#ifndef PARSING_H
#define PARSING_H

#include <cstdlib>
#include <optional>
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
    const std::string flag = argv[i];

    // See if the key is recognized
    auto itr = std::find_if(dict.begin(), dict.end(),
                            [&flag](pair<string, T *> &dict_pair) {
                              return (flag == dict_pair.first);
                            });

    // If it is, and the specification is complete
    // remove the dict entry
    if (itr != dict.end()) {
      if (i + 1 < argc) {
        string value = string(argv[++i]);
        std::optional<T> processed = convertor(value);
        if (!processed) {
          printf("value %s requested for flag %s is not recognized.\n",
                 value.c_str(), flag.c_str());
          std::exit(EXIT_FAILURE);
        }

        *(itr->second) = processed.value();
        dict.erase(itr);
        printf("processed (%s, %s) pair.\n", flag.c_str(), value.c_str());
      } else {
        printf("value missing for specified flag %s.\n", flag.c_str());
        std::exit(EXIT_FAILURE);
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

static std::optional<int> int_from_string(const string &str) {
  return std::stoi(str);
}
static std::optional<double> double_from_string(const string &str) {
  return std::stod(str);
}
static std::optional<string> string_from_string(const string &str) {
  return str;
}

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