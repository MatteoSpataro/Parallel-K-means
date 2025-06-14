#ifdef _OPENMP
#include <omp.h> // for OpenMP library functions
#endif
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <tuple>

using namespace std;

// cl /std:c++20 /openmp:llvm /O2 Kmeans.cpp /Fe:kmeans.exe /EHsc
// kmeans.exe big_five_personality.csv -p 5 1000 24

const double EPSILON = 1e-6;

// Read dataset from CSV file assuming that the first row contains the feature names
// and all the values are numbers.
tuple<vector<vector<double>>, vector<string>> readDataset(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file " + filename);
    }

    vector<vector<double>> data;
    vector<string> headers;
    string line;
    if (getline(file, line)) {
        size_t start = 0;
        size_t end = line.find(',');
        while (end != string::npos) {
            headers.emplace_back(line.substr(start, end - start));
            start = end + 1;
            end = line.find(',', start);
        }
        headers.emplace_back(line.substr(start));
    }

    while (getline(file, line)) {
        vector<double> row;
        size_t start = 0;
        size_t end = line.find(',');
        bool valid = true;

        while (end != string::npos) {
            string token = line.substr(start, end - start);
            try {
                if (!token.empty())
                    row.push_back(stod(token));
                else {
                    valid = false;
                    break;
                }
            }
            catch (const std::invalid_argument& e) {
                valid = false;
                cerr << "Invalid argument while parsing token " << token << ": " << e.what() << endl;
                break;
            }
            catch (const std::out_of_range& e) {
                valid = false;
                cerr << "Out of range while parsing token " << token << ": " << e.what() << endl;
                break;
            }
            start = end + 1;
            end = line.find(',', start);
        }

        string token = line.substr(start);
        try {
            if (!token.empty())
                row.push_back(stod(token));
            else
                valid = false;
        }
        catch (const std::invalid_argument& e) {
            valid = false;
            cerr << "Invalid argument while parsing token " << token << ": " << e.what() << endl;
        }
        catch (const std::out_of_range& e) {
            valid = false;
            cerr << "Out of range while parsing token " << token << ": " << e.what() << endl;
        }

        if (valid && row.size() == headers.size()) {
            data.emplace_back(move(row));
        }
        else if (valid) {
            cerr << "At line " << line << ", row has incorrect number of columns: headers.size() = "
                << headers.size() << ", row.size() = " << row.size() << endl;
        }
    }
    return { move(data), move(headers) };
}

double squareDistance(const vector<double>& x, const vector<double>& y) {
    double dist = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double diff = x[i] - y[i];
        dist += diff * diff;
    }
    return dist;
}

vector<vector<double>> kmeans(const vector<vector<double>>& data, int K, 
    int maxIterations) {
    vector<vector<double>> centroids;
    for (int i = 0; i < K; ++i) {
        centroids.push_back(data[i]);
    }

    int iteration = 0;
    double maxChange;
    do {
        vector<vector<double>> sums(K, vector<double>(data[0].size(), 0.0));
        vector<int> counts(K, 0);

        for (const auto& point : data) {
            double min_dist = numeric_limits<double>::max();
            int cluster = -1;

            for (int j = 0; j < K; ++j) {
                double dist = squareDistance(point, centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster = j;
                }
            }

            if (cluster != -1) {
                for (size_t d = 0; d < point.size(); ++d) {
                    sums[cluster][d] += point[d];
                }
                counts[cluster]++;
            }
        }

        maxChange = 0.0;
        for (int j = 0; j < K; ++j) {
            vector<double> old = centroids[j];
            if (counts[j] > 0) {
                for (size_t d = 0; d < centroids[j].size(); ++d) {
                    centroids[j][d] = sums[j][d] / counts[j];
                }
            }
            double change = squareDistance(old, centroids[j]);
            if (change > maxChange) {
                maxChange = change;
            }
        }

        iteration++;
    } while (maxChange > EPSILON && iteration < maxIterations);
    std::cout << "\nEnd after " << iteration << " iterations." << std::endl;
    return centroids;
}


// Parallel K-means using OpenMP
vector<vector<double>> kmeansParallel(const vector<vector<double>>& data, int K,
    int maxIterations) {
    vector<vector<double>> centroids;
    for (int i = 0; i < K; ++i) {
        centroids.push_back(data[i]);
    }

    int iteration = 0;
    double maxChange;
    const int n = data.size();
    const int d = data[0].size();

    do {
        vector<double> flatSums(K * d, 0.0);  // Global sums 
        vector<int> counts(K, 0);             // Global counts

#pragma omp parallel
        {
            vector<double> localFlatSums(K * d);
            vector<int> localCounts(K, 0);
            
            // Parallel First touch
#pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < K * d; ++i){
                localFlatSums[i] = 0.0;
            }
#pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < n; i++) {
                const vector<double>& point = data[i];
                double min_dist = numeric_limits<double>::max();
                int cluster = -1;

                // Find nearest centroid
                for (int j = 0; j < K; ++j) {
                    double dist = squareDistance(point, centroids[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        cluster = j;
                    }
                }

                if (cluster != -1) {
                    for (int dim = 0; dim < d; dim++) {
                        localFlatSums[cluster * d + dim] += point[dim];
                    }
                    localCounts[cluster]++;
                }
            }

            for (int j = 0; j < K; j++) {
#pragma omp atomic
                counts[j] += localCounts[j];
            }
            for (int idx = 0; idx < K * d; idx++) {
#pragma omp atomic
                flatSums[idx] += localFlatSums[idx];
            }

        }
        maxChange = 0.0;
#pragma omp parallel for reduction(max:maxChange) schedule(dynamic)
        for (int j = 0; j < K; j++) {
            vector<double> old = centroids[j];
            if (counts[j] > 0) {
                // Update centroid coordinates
                for (int dim = 0; dim < d; dim++) {
                    centroids[j][dim] = flatSums[j * d + dim] / counts[j];
                }
            }
            double change = squareDistance(old, centroids[j]);
            maxChange = std::max(maxChange, change);
        }

        iteration++;
    } while (maxChange > EPSILON && iteration < maxIterations);
    std::cout << "\nEnd after " << iteration << " iterations." << std::endl;
    return centroids;
}

void writeResults(const vector<vector<double>>& data,
    const vector<vector<double>>& centroids, int K,
    const vector<string>& featureNames, int numFeatures) {
    ofstream output("clustered_data.csv");
    // Write header
    if (!featureNames.empty()) {
        for (size_t i = 0; i < featureNames.size(); ++i) {
            output << featureNames[i] << ",";
        }
    }
    else {
        for (size_t i = 0; i < numFeatures; ++i) {
            output << "Feature_" << (i + 1) << ",";
        }
    }
    output << "cluster\n";

    // Write data
    for (const auto& point : data) {
        int cluster = 0;
        double min_dist = numeric_limits<double>::max();
        for (int j = 0; j < K; ++j) {
            double dist = squareDistance(point, centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                cluster = j;
            }
        }
        for (auto value : point) {
            output << value << ",";
        }
        output << cluster << "\n";
    }
    output.close();
}

void writeCentroids(const vector<vector<double>>& centroids, 
    const vector<string>& featureNames, int numFeatures) {
    ofstream output("centroids.csv");
    // Write header
    if (!featureNames.empty()) {
        for (size_t i = 0; i < featureNames.size(); ++i) {
            output << featureNames[i];
            if (i != featureNames.size() - 1) output << ",";
        }
    }
    else {
        for (size_t i = 0; i < numFeatures; ++i) {
            output << "Feature_" << (i + 1);
            if (i != numFeatures - 1) output << ",";
        }
    }
    output << "\n";

    // Write centroids
    for (const auto& centroid : centroids) {
        if (centroid.empty()) continue;
        for (size_t i = 0; i < centroid.size(); ++i) {
            output << centroid[i];
            if (i != centroid.size() - 1) output << ",";
        }
        output << "\n";
    }
    output.close();
}

int run(int argc, char* argv[]) {
    using clock = chrono::high_resolution_clock;
    auto start = clock::now();
    // Expect dataset filename (without path) as argument
    if (argc < 6) {
        cerr << "The input parameters are:"
            << endl;
        cerr << "<dataset.csv> <-p or -s> <K> <maxIterations> <threads>" << endl;
        cerr << "Remember that dataset must be located in ./datasets directory." << endl;
        return 1;
    }
    int threads = 12;
    sscanf(argv[5], "%d", &threads);
    omp_set_num_threads(threads);
    string dataset_name = argv[1];
    string filename = "./datasets/" + dataset_name;
    int K = 5;
    sscanf(argv[3], "%d", &K);
    std::cout << "Number of centroids = " << K << endl;
    int maxIterations = 1000;
    sscanf(argv[4], "%d", &maxIterations);
    std::cout << "Number of max iterations = " << maxIterations << endl;
    if (string(argv[2]) == "-p") {
        std::cout << "Number of threads = " << threads << endl;
    }
    vector<vector<double>> data;
    vector<string> featureNames;

    try {
        tie(data, featureNames) = readDataset(filename);
        std::cout << "Rows: " << data.size() << "; Columns: " << featureNames.size() << endl;
    }
    catch (const ifstream::failure& e) {
        cerr << "I/O error while reading the file " << filename << ": " << e.what() << endl;
        return 1;
    }
    catch (const exception& e) {
        cerr << "Error while processing dataset: " << e.what() << endl;
        return 1;
    }
    if (data.empty()) {
        cerr << "Error: the dataset is empty." << endl;
        return 1;
    }
    if (data.size() < K) {
        cerr << "Error: Dataset contains fewer than " << K << "points ("
            << data.size() << endl;
        return 1;
    }


    size_t numFeatures = data[0].size();
    // Check integrity of each row  
    for (const auto& point : data) {
        if (point.size() != numFeatures) {
            cerr << "Error: Inconsistent number of features in dataset" << endl;
            return 1;
        }
    }
    vector<vector<double>> centroids;
    auto end = clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nTime spent to read the data: " << elapsed.count() << " seconds." << std::endl;

    start = clock::now();
    if (string(argv[2]) == "-p") {
        std::cout << "Using " << omp_get_max_threads() << " threads." << endl;
        centroids = kmeansParallel(data, K, maxIterations);
    }
    else {
        centroids = kmeans(data, K, maxIterations);
    }
    end = clock::now();
    elapsed = end - start;
    std::cout << "\nTime spent to cluster the data: " << elapsed.count() << " seconds." << std::endl;
    /*
    // Output final centroids with feature names
    std::cout << "\nFinal centroids:" << endl;
    for (int i = 0; i < K; ++i) {
        std::cout << "\nCluster " << i << ":\n";
        for (size_t j = 0; j < centroids[i].size(); ++j) {
            std::cout << "  " << featureNames[j] << ": " << centroids[i][j] << endl;
        }
    }
    */
    /*
    start = clock::now();
    // Save clustered data with headers
    writeResults(data, centroids, featureNames, numFeatures);
    end = clock::now();
    elapsed = end - start;
    std::cout << "\nTime spent to write the data: " << elapsed.count() << " seconds." << std::endl;
    */
    // Save centroids with headers
    //writeCentroids(centroids, featureNames, numFeatures);
    return 0;
}

int main(int argc, char* argv[]) {
    int N = 10;
    for (size_t i = 0; i < N; i++) {
        int ret = run(argc, argv);
        if (ret == 1) {
            cerr << "Error occured at run number " << i+1 << endl;
            return 1;
        }
    }    
    return 0;
}
