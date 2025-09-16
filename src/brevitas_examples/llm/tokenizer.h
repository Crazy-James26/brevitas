#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cctype>

#include "parameters/embedding_look_up.h" // VOCAB

static bool is_punct(char c) {
    static const std::string puncts = ".,?!\"'$!:;()[]{}";
    return puncts.find(c) != std::string::npos;
}

std::vector<std::string> split_with_punct(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : text) {
        if (std::isspace((unsigned char)c)) {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        } else if (is_punct(c)) {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
            tokens.emplace_back(1, c); 
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

std::vector<int> tokenize(const std::string& text, bool add_bos=true, bool add_eos=true) {
    std::vector<int> ids;
    if (add_bos) ids.push_back(VOCAB.at("<s>")); // BOS

    auto tokens = split_with_punct(text);
    cout << "Tokens: ";
    for (const auto& t : tokens) cout << "[" << t << "] ";
    
    for (auto& word : tokens) {
        auto it = VOCAB.find(word);
        if (it != VOCAB.end()) {
            ids.push_back(it->second);
        } else {
            if (word.size() > 1 && word.back() == 's') {
                std::string stem = word.substr(0, word.size() - 1);
                auto it_stem = VOCAB.find(stem);
                auto it_s    = VOCAB.find("s");
                if (it_stem != VOCAB.end() && it_s != VOCAB.end()) {
                    ids.push_back(it_stem->second);
                    ids.push_back(it_s->second);
                    continue;
                }
            }
            
            std::cerr << "Unknown token: " << word << "\n";
            ids.push_back(VOCAB.at("▁")); // fallback UNK
        }
    }

    if (add_eos) ids.push_back(VOCAB.at("</s>")); // EOS
    return ids;
}


// convenience wrapper that returns raw int*
int* tokenize_to_array(const std::string& text, int& out_len) {
    std::vector<int> ids = tokenize(text);
    out_len = (int)ids.size();
    int* arr = new int[out_len];
    for (int i = 0; i < out_len; i++) arr[i] = ids[i];
    return arr;
}

