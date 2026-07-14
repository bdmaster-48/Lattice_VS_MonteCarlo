// C++ code by Claude
// -----------------------------------------------------------------------
// 목적: 결정적 수치적분(격자 기반 사다리꼴)과 몬테카를로 적분을
//       차원(d)을 늘려가며 비교하여 "차원의 저주"를 실험적으로 확인.
//
// 대상 함수: f(x1,...,xd) = exp(-x1^2) * exp(-x2^2) * ... * exp(-xd^2)
//            적분 구간: [0,1]^d  (각 축이 독립이므로 참값을 해석적으로 구할 수 있음)
//
// 참값(analytic value):
//   1차원 적분 I1 = ∫0^1 exp(-x^2) dx = (sqrt(pi)/2) * erf(1)  (수치적으로 계산)
//   d차원 적분 Id = I1^d   (각 축이 독립인 곱 형태이므로)
//
// 실험 1: 격자 기반 사다리꼴 적분 -> 차원이 늘어날수록 필요한 격자점 수가
//         n^d 로 폭증하여 계산이 사실상 불가능해짐을 보여줌.
// 실험 2: 몬테카를로 적분 -> 표본 수 N에 대해 오차가 O(1/sqrt(N))로
//         차원과 무관하게 감소함을 보여줌.
// -----------------------------------------------------------------------

#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;
long double pi=acos(-1);
// 피적분함수의 1차원 성분: exp(-x^2)
static inline double f1d(double x) {
    return exp(-x * x);
}

// ---------------------------------------------------------------------
// 1) 1차원 사다리꼴 공식으로 I1 = ∫0^1 exp(-x^2) dx 근사
// ---------------------------------------------------------------------
double trapezoid1D(int n) {
    double a = 0.0, b = 1.0;
    double h = (b - a) / n;
    double sum = 0.5 * (f1d(a) + f1d(b));
    for (int i = 1; i < n; i++) {
        sum += f1d(a + i * h);
    }
    return sum * h;
}

// ---------------------------------------------------------------------
// 2) d차원 격자 기반 사다리꼴 적분
//    각 축에 n개의 격자 구간(n+1개의 점)을 사용 -> 총 (n+1)^d 번 함수 평가
//    d가 커지면 total 이 즉시 감당 불가능한 수준으로 커지므로,
//    total 이 안전 상한(MAX_GRID_EVALS)을 넘으면 "계산 불가"로 처리한다.
// ---------------------------------------------------------------------
const long long MAX_GRID_EVALS = 200'000'000LL; // 이 이상은 시간/메모리상 사실상 불가능

struct GridResult {
    bool feasible;
    double value;
    long long evals;
    double seconds;
};

GridResult trapezoidND(int d, int n) {
    // 총 함수 평가 횟수 = (n+1)^d
    long long total = 1;
    for (int i = 0; i < d; i++) {
        total *= (n + 1);
        if (total > MAX_GRID_EVALS) {
            return {false, 0.0, total, 0.0};
        }
    }

    double h = 1.0 / n;

    // 1차원 사다리꼴 가중치(끝점 0.5, 내부점 1.0)를 미리 계산해 곱으로 결합
    vector<double> w(n + 1), x(n + 1);
    for (int i = 0; i <= n; i++) {
        x[i] = i * h;
        w[i] = (i == 0 || i == n) ? 0.5 : 1.0;
    }

    auto start = high_resolution_clock::now();

    // d차원 텐서곱 격자를 혼합진법(mixed-radix) 카운터로 순회
    vector<int> idx(d, 0);
    double sum = 0.0;
    while (true) {
        double weight = 1.0, fval = 1.0;
        for (int k = 0; k < d; k++) {
            weight *= w[idx[k]];
            fval *= f1d(x[idx[k]]);
        }
        sum += weight * fval;

        // 카운터 증가 (odometer 방식)
        int pos = 0;
        while (pos < d) {
            idx[pos]++;
            if (idx[pos] <= n) break;
            idx[pos] = 0;
            pos++;
        }
        if (pos == d) break; // 모든 자리가 넘침 -> 순회 종료
    }

    double value = sum * pow(h, d);
    auto end = high_resolution_clock::now();
    double sec = duration<double>(end - start).count();

    return {true, value, total, sec};
}

// ---------------------------------------------------------------------
// 3) d차원 몬테카를로 적분
//    [0,1]^d 에서 균등분포로 N개의 점을 무작위 추출하여 함수값의 평균을 구함
// ---------------------------------------------------------------------
struct MCResult {
    double value;
    double stderr_; // 표준오차 (표본표준편차 / sqrt(N))
    double seconds;
};

MCResult monteCarloND(int d, long long N, unsigned seed) {
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);

    auto start = high_resolution_clock::now();

    double sum = 0.0, sumSq = 0.0;
    vector<double> x(d);
    for (long long i = 0; i < N; i++) {
        double fval = 1.0;
        for (int k = 0; k < d; k++) fval *= f1d(U(rng));
        sum += fval;
        sumSq += fval * fval;
    }
    double mean = sum / N;
    double var = sumSq / N - mean * mean;
    double se = sqrt(max(var, 0.0) / N); // O(1/sqrt(N))

    auto end = high_resolution_clock::now();
    double sec = duration<double>(end - start).count();

    return {mean, se, sec};
}

int main() 
{
    system("chcp 65001 > nul");
    cout << fixed;

    // 참값 계산: I1 = (sqrt(pi)/2) * erf(1),  Id = I1^d
    double I1 = (sqrt(pi) / 2.0) * erf(1.0);
    cout << "1차원 참값 I1 = " << setprecision(10) << I1 << "\n\n";

    cout << "=====================================================\n";
    cout << " 실험 1. 격자 기반 사다리꼴 적분 (차원의 저주 확인)\n";
    cout << "=====================================================\n";
    cout << left << setw(6) << "d" << setw(10) << "n(축당)"
         << setw(16) << "총 평가 횟수" << setw(14) << "근사값"
         << setw(14) << "절대오차" << setw(10) << "시간(s)" << "\n";

    int nPerAxis = 50; // 축당 격자 구간 수 (정확도 확보를 위해 고정)
    for (int d = 1; d <= 8; d++) {
        double trueVal = pow(I1, d);
        GridResult r = trapezoidND(d, nPerAxis);
        cout << setw(6) << d << setw(10) << nPerAxis;
        if (r.feasible) {
            double err = fabs(r.value - trueVal);
            cout << setw(16) << r.evals
                 << setw(14) << setprecision(6) << r.value
                 << setw(14) << setprecision(2) << scientific << err << fixed
                 << setw(10) << setprecision(3) << r.seconds << "\n";
        } else {
            cout << setw(16) << r.evals << setw(14) << "계산불가" << setw(14) << "-" << setw(10) << "-" << "\n";
        }
    }

    cout << "\n=====================================================\n";
    cout << " 실험 2. 몬테카를로 적분 (차원과 무관한 O(1/sqrt(N)) 수렴)\n";
    cout << "=====================================================\n";
    cout << left << setw(6) << "d" << setw(14) << "표본 수 N"
         << setw(14) << "근사값" << setw(14) << "절대오차"
         << setw(14) << "표준오차" << setw(10) << "시간(s)" << "\n";

    vector<long long> sampleCounts = {1000, 10000, 100000, 1000000};
    for (int d = 1; d <= 8; d++) {
        double trueVal = pow(I1, d);
        for (long long N : sampleCounts) {
            MCResult r = monteCarloND(d, N, /*seed=*/12345u + d * 100000u + (unsigned)N);
            double err = fabs(r.value - trueVal);
            cout << setw(6) << d << setw(14) << N
                 << setw(14) << setprecision(6) << r.value
                 << setw(14) << setprecision(2) << scientific << err << fixed
                 << setw(14) << setprecision(2) << scientific << r.stderr_ << fixed
                 << setw(10) << setprecision(3) << r.seconds << "\n";
        }
        cout << "-----------------------------------------------------\n";
    }

    return 0;
}
