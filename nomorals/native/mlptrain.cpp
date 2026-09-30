// mlptrain.cpp — C ABI kernel for the native next-token MLP trainer.
//
// The phone's training backend (nomorals/training/trainer.py) is pure
// Python: every (context, target) pair pays interpreter overhead on each
// multiply-add.  This kernel runs the exact same math — same operation
// order, same clamps, same accumulation order — in C++ so a batch of
// samples is one native call.  The Python implementation is the
// reference: tests run both and compare loss + every gradient element.
//
// Build: c++ -O2 -fPIC -shared -std=c++17 mlptrain.cpp -o libmlptrain.so
// (Termux clang on the phone, g++/clang++ on desktop — same flags as
// vecsim.cpp; no dependencies beyond libstdc++.)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" {

const char* nm_mlp_version() { return "1.0.0"; }

// One training (or evaluation, lr == 0) batch.
//
//   seqs/lens  — n token sequences (packed; lens[i] valid entries)
//   embedding  — [vocab][hidden]   (const unless lr != 0, then updated)
//   hidden_w   — [hidden][hidden]  (same)
//   hidden_b   — [hidden]          (same)
//   out_w      — [hidden][vocab]   (same)
//   out_b      — [vocab]           (same)
//   lr         — step size applied as  w += -lr*grad - l2*lr*w, b += -lr*grad
//   l2         — weight-decay factor (already includes nothing; python
//                passes config.l2 * lr, exactly like _axpy does)
//   grad_*     — scratch, zeroed here; after the call they hold the
//                batch-mean gradients (divided by count), ready to
//                compare against the Python reference
//   count_out  — number of (context, target) pairs processed
//
// Returns the batch-mean negative log-likelihood (0.0 when count == 0).
double nm_mlp_batch(
    const int32_t* seqs, const int32_t* lens, int32_t n,
    double* embedding, double* hidden_w, double* hidden_b,
    double* out_w, double* out_b,
    int32_t vocab, int32_t hidden, int32_t context,
    double lr, double l2,
    double* grad_embed,    // [vocab][hidden]
    double* grad_hidden_w, // [hidden][hidden]
    double* grad_hidden_b, // [hidden]
    double* grad_out_w,    // [hidden][vocab]
    double* grad_out_b,    // [vocab]
    int32_t* count_out)
{
    const size_t vh = (size_t)vocab * (size_t)hidden;
    const size_t hh = (size_t)hidden * (size_t)hidden;
    const size_t hv = (size_t)hidden * (size_t)vocab;

    // Zero the gradient accumulators.
    for (size_t i = 0; i < vh; i++) grad_embed[i] = 0.0;
    for (size_t i = 0; i < hh; i++) grad_hidden_w[i] = 0.0;
    for (size_t i = 0; i < (size_t)hidden; i++) grad_hidden_b[i] = 0.0;
    for (size_t i = 0; i < hv; i++) grad_out_w[i] = 0.0;
    for (size_t i = 0; i < (size_t)vocab; i++) grad_out_b[i] = 0.0;

    std::vector<double> ctx((size_t)hidden);
    std::vector<double> pre((size_t)hidden);
    std::vector<double> post((size_t)hidden);
    std::vector<double> grad_hidden((size_t)hidden);
    std::vector<double> logits((size_t)vocab);
    std::vector<double> exps((size_t)vocab);
    std::vector<double> log_probs((size_t)vocab);
    std::vector<double> probs((size_t)vocab);

    double total_loss = 0.0;
    long count = 0;

    for (int32_t s = 0; s < n; s++) {
        const int32_t* seq = seqs;
        for (int32_t i = 0; i < s; i++) seq += lens[i];
        const int32_t len = lens[s];
        if (len <= context) continue;

        for (int32_t position = context; position < len; position++) {
            const int32_t target = seq[position];

            // Context vector: mean of the last `context` token embeddings
            // (mirror of NativeTrainer._context_vector; window is exactly
            // `context` long because position >= context).
            const int32_t wstart = position - context;
            for (int32_t j = 0; j < hidden; j++) ctx[j] = 0.0;
            for (int32_t t = wstart; t < position; t++) {
                const double* row = embedding + (size_t)(seq[t] % vocab) * (size_t)hidden;
                for (int32_t j = 0; j < hidden; j++) ctx[j] += row[j];
            }
            const double factor = 1.0 / (double)context;
            for (int32_t j = 0; j < hidden; j++) ctx[j] *= factor;

            // Forward: hidden = ReLU(Hw @ ctx + hb); logits = post @ OutW + ob
            for (int32_t j = 0; j < hidden; j++) {
                const double* row = hidden_w + (size_t)j * (size_t)hidden;
                double total = hidden_b[j];
                for (int32_t k = 0; k < hidden; k++) total += ctx[k] * row[k];
                pre[j] = total;
                post[j] = (total > 0.0) ? total : 0.0;
            }
            for (int32_t i = 0; i < vocab; i++) logits[i] = out_b[i];
            for (int32_t j = 0; j < hidden; j++) {
                if (post[j] == 0.0) continue;
                const double* row = out_w + (size_t)j * (size_t)vocab;
                for (int32_t i = 0; i < vocab; i++) logits[i] += post[j] * row[i];
            }

            // Numerically stable log-softmax, same clamps as the python.
            double maximum = logits[0];
            for (int32_t i = 1; i < vocab; i++) if (logits[i] > maximum) maximum = logits[i];
            double sumexp = 0.0;
            for (int32_t i = 0; i < vocab; i++) {
                double d = logits[i] - maximum;
                if (d > 50.0) d = 50.0;
                exps[i] = std::exp(d);
                sumexp += exps[i];
            }
            for (int32_t i = 0; i < vocab; i++) {
                double r = exps[i] / sumexp;
                if (r < 1e-12) r = 1e-12;
                log_probs[i] = std::log(r);
            }

            total_loss -= log_probs[target];
            count += 1;

            // probs = exp(clamp(log_probs, -, 50)) — the python's
            // `probs = [math.exp(min(50.0, lp)) for lp in log_probs]`.
            for (int32_t i = 0; i < vocab; i++) {
                double lp = log_probs[i];
                if (lp > 50.0) lp = 50.0;
                probs[i] = std::exp(lp);
            }

            // Output gradients: dL/d(OutW) = post ⊗ probs, minus one-hot.
            for (int32_t j = 0; j < hidden; j++) {
                if (post[j] == 0.0) continue;
                double* row = grad_out_w + (size_t)j * (size_t)vocab;
                for (int32_t i = 0; i < vocab; i++) row[i] += post[j] * probs[i];
                row[target] -= post[j];
            }
            for (int32_t i = 0; i < vocab; i++) grad_out_b[i] += probs[i];
            grad_out_b[target] -= 1.0;

            // Backprop through ReLU into the hidden layer and embeddings.
            // (Mirror of the python: grad_hidden per column first, then the
            // window tokens each receive grad_hidden[jj] for their column jj.)
            for (int32_t j = 0; j < hidden; j++) grad_hidden[j] = 0.0;
            for (int32_t j = 0; j < hidden; j++) {
                if (pre[j] <= 0.0) continue;
                const double* owrow = out_w + (size_t)j * (size_t)vocab;
                double acc = 0.0;
                for (int32_t i = 0; i < vocab; i++) {
                    double delta = probs[i] - (i == target ? 1.0 : 0.0);
                    acc += delta * owrow[i];
                }
                grad_hidden[j] = acc;
                grad_hidden_b[j] += acc;
                double* row = grad_hidden_w + (size_t)j * (size_t)hidden;
                for (int32_t k = 0; k < hidden; k++) row[k] += acc * ctx[k];
            }
            for (int32_t t = wstart; t < position; t++) {
                double* erow = grad_embed + (size_t)(seq[t] % vocab) * (size_t)hidden;
                for (int32_t jj = 0; jj < hidden; jj++) erow[jj] += grad_hidden[jj] * factor;
            }
        }
    }

    const double divisor = (count > 0) ? (double)count : 1.0;
    for (size_t i = 0; i < vh; i++) grad_embed[i] /= divisor;
    for (size_t i = 0; i < hh; i++) grad_hidden_w[i] /= divisor;
    for (size_t i = 0; i < (size_t)hidden; i++) grad_hidden_b[i] /= divisor;
    for (size_t i = 0; i < hv; i++) grad_out_w[i] /= divisor;
    for (size_t i = 0; i < (size_t)vocab; i++) grad_out_b[i] /= divisor;

    if (lr != 0.0) {
        // SGD step with weight decay — the python _axpy/_add, in place:
        //   w = w - lr*g - (l2*lr)*w      (l2 already passed as l2*lr)
        //   b = b - lr*g
        for (size_t i = 0; i < vh; i++)
            embedding[i] = embedding[i] - lr * grad_embed[i] - l2 * embedding[i];
        for (size_t i = 0; i < hh; i++)
            hidden_w[i] = hidden_w[i] - lr * grad_hidden_w[i] - l2 * hidden_w[i];
        for (size_t i = 0; i < (size_t)hidden; i++)
            hidden_b[i] = hidden_b[i] - lr * grad_hidden_b[i];
        for (size_t i = 0; i < hv; i++)
            out_w[i] = out_w[i] - lr * grad_out_w[i] - l2 * out_w[i];
        for (size_t i = 0; i < (size_t)vocab; i++)
            out_b[i] = out_b[i] - lr * grad_out_b[i];
    }

    if (count_out) *count_out = (int32_t)count;
    return (count > 0) ? total_loss / (double)count : 0.0;
}

} // extern "C"
