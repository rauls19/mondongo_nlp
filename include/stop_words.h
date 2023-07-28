#ifndef SW_LANG
#define SW_LANG

#include <unordered_set>
#include <map>
#include<string>

using namespace std;

extern const unordered_set<string> sw_en;
extern const unordered_set<string> sw_es;
extern map<string, const unordered_set<string>*> sw_lang_map;

#endif