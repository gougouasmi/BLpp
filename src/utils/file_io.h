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

#endif