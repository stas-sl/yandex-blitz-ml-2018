/*
  This is naive O(N*N) solution, which results in Time Limit Exceeded,
  but can be used on small inputs to check correctness
*/

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main () {
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    int n;
    cin >> n;
    vector<pair<double, double>> a(n);
    for (int i = 0; i < n; i ++) {
        cin >> a[i].first;
        cin >> a[i].second;
    }

    double nom = 0, denom = 0;
    for (int i = 0; i < n; i ++) 
        for (int j = 0; j < n; j ++)
            if (a[i].first > a[j].first) {
                denom += 1;
                if (a[i].second > a[j].second)
                    nom += 1;
                else if (a[i].second == a[j].second)
                    nom += 0.5;
            }

    cout << nom / denom;
}