// Read-only compatibility oracle; compile with the sibling config.cpp/orbit.cpp.
#include <distribution/config.hpp>
#include <iomanip>
#include <iostream>
#include <random>
int main() {
    std::mt19937_64 generator(5489);
    for (int i=0; i<1000; ++i) {
        const auto value=generator();
        if(i<10 || i==311 || i==312 || i==623 || i==624 || i==999)
            std::cout << "rng," << i << ',' << value << '\n';
    }
    distribution::Config c;
    c.samples=5;
    c.threads=1;
    std::cout << std::setprecision(17);
    for(const auto& row: distribution::sample_initial(c)) {
        std::cout << "sample";
        for(const auto value:row) std::cout << ',' << value;
        std::cout << '\n';
    }
}
