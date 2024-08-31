#ifndef FILE_IO_H
#define FILE_IO_H

#include <string>
#include <vector>

using std::string;
using std::vector;

void GetDimsCSV(const string &file_path, int &nb_rows, int &nb_cols);
vector<vector<double>> ReadCSV(const string &file_path);
void WriteCSV(const string &file_path,
              const vector<vector<double>> &data_columns);
void WriteCSV(const string &file_path, const vector<double> &data, int rank,
              int nb_points);
void WriteCSV(const string &file_path, const vector<double> &data, int rank,
              int nb_points, const vector<int> &var_indices);

#endif