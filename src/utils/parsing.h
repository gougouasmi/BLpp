#ifndef PARSING_H
#define PARSING_H

#include <string>
#include <utility>
#include <vector>

using std::pair;
using std::string;
using std::vector;

template <typename T, typename StringToTFunction>
void ParseValues(int argc, char *argv[], vector<pair<string, T *>> dict,
                 StringToTFunction convertor) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    // See if the key is recognized
    auto itr = std::find_if(dict.begin(), dict.end(),
                            [&arg](pair<string, T *> &dict_pair) {
                              return (arg == dict_pair.first);
                            });

    // If it is, and the specification is complete
    // remove the dict entry
    if (itr != dict.end()) {
      if (i + 1 < argc) {
        string value = string(argv[++i]);
        *(itr->second) = convertor(value);
        dict.erase(itr);
        printf("processed (%s, %s) pair.\n", arg.c_str(), value.c_str());
      } else {
        printf("value missing for command line argument %s.\n", arg.c_str());
      }
    }
  }
}

static inline void ParseOptions(int argc, char *argv[],
                                vector<pair<string, bool *>> dict) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    // See if the key is recognized
    auto itr = std::find_if(dict.begin(), dict.end(),
                            [&arg](pair<string, bool *> &dict_pair) {
                              return (arg == dict_pair.first);
                            });

    // If it is, and the specification is complete
    // remove the dict entry
    if (itr != dict.end()) {
      *(itr->second) = true;
      dict.erase(itr);
      printf("processed %s option.\n", arg.c_str());
    }
  }
}

static int int_from_string(string &str) { return std::stoi(str); }
static double double_from_string(string &str) { return std::stod(str); }
static string string_from_string(string &str) { return str; }

static inline void ParseValues(int argc, char *argv[],
                               vector<pair<string, int *>> dict) {
  ParseValues(argc, argv, dict, int_from_string);
}

static inline void ParseValues(int argc, char *argv[],
                               vector<pair<string, double *>> dict) {
  ParseValues(argc, argv, dict, double_from_string);
}

static inline void ParseValues(int argc, char *argv[],
                               vector<pair<string, string *>> dict) {
  ParseValues(argc, argv, dict, string_from_string);
}

#endif