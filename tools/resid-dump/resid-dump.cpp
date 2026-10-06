// llama-resid-dump: the residual stream after every layer (the l_out tensors) at the LAST token of each prompt.
//
//   llama-resid-dump -m model.gguf --positive-file prompts.txt -o out.bin -ngl 99 -c 2048
//
// prompts.txt: one prompt per line, already in the chat template, "\n" escapes allowed, special tokens parsed.
// out.bin: float32 [n_prompts, n_layer, n_embd] in prompt order; out.bin.json has the shape.
// Built from tools/cvector-generator (same eval callback), without the pairing and the PCA.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct dump_state {
    int n_tokens = 0;
    int n_layer  = 0;
    int n_embd   = 0;
    std::vector<float> rows;   // n_layer * n_embd of the current prompt
    std::vector<char>  seen;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const bool is_l_out = strncmp(t->name, "l_out-", 6) == 0;
    if (ask) {
        return is_l_out;
    }
    if (!is_l_out || t->ne[1] != st->n_tokens || t->ne[0] != st->n_embd || t->type != GGML_TYPE_F32) {
        return true;
    }
    const int il = atoi(t->name + 6);
    if (il < 0 || il >= st->n_layer) {
        return true;
    }
    const size_t row = (size_t) st->n_embd * sizeof(float);
    ggml_backend_tensor_get(t, st->rows.data() + (size_t) il * st->n_embd, (size_t) (st->n_tokens - 1) * row, row);
    st->seen[il] = 1;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    params.out_file = "resid.bin";
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CVECTOR_GENERATOR)) {
        return 1;
    }
    dump_state st;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    llama_backend_init();
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }
    st.n_layer = llama_model_n_layer(model);
    st.n_embd  = llama_model_n_embd(model);
    st.rows.resize((size_t) st.n_layer * st.n_embd);
    st.seen.resize(st.n_layer);

    std::ifstream in(params.cvector_positive_file);
    if (!in.is_open()) {
        fprintf(stderr, "cannot open %s\n", params.cvector_positive_file.c_str());
        return 1;
    }
    FILE * out = fopen(params.out_file.c_str(), "wb");
    if (out == nullptr) {
        fprintf(stderr, "cannot write %s\n", params.out_file.c_str());
        return 1;
    }
    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    std::string line;
    int n_prompts = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        string_process_escapes(line);
        std::vector<llama_token> tokens = common_tokenize(ctx, line, add_bos, true);
        st.n_tokens = (int) tokens.size();
        std::fill(st.seen.begin(), st.seen.end(), 0);
        llama_memory_clear(llama_get_memory(ctx), true);
        if (st.n_tokens > (int) llama_n_batch(ctx) || llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
            fprintf(stderr, "prompt %d: failed to eval (%d tokens)\n", n_prompts, st.n_tokens);
            return 1;
        }
        int n_seen = 0;
        for (char s : st.seen) n_seen += s;
        if (n_seen != st.n_layer) {
            fprintf(stderr, "prompt %d: %d of %d layers captured\n", n_prompts, n_seen, st.n_layer);
            return 1;
        }
        fwrite(st.rows.data(), sizeof(float), st.rows.size(), out);
        n_prompts++;
        if (n_prompts % 20 == 0) {
            fprintf(stderr, "resid-dump: %d prompts, last %d tokens\n", n_prompts, st.n_tokens);
        }
    }
    fclose(out);
    FILE * meta = fopen((params.out_file + ".json").c_str(), "w");
    fprintf(meta, "{\"n_prompts\": %d, \"n_layer\": %d, \"n_embd\": %d, \"dtype\": \"float32\", \"position\": \"last token\"}\n",
            n_prompts, st.n_layer, st.n_embd);
    fclose(meta);
    fprintf(stderr, "resid-dump: wrote %d prompts x %d layers x %d to %s\n", n_prompts, st.n_layer, st.n_embd, params.out_file.c_str());
    llama_backend_free();
    return 0;
}
