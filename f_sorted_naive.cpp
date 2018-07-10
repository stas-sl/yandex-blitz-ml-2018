/*
  This is a bit better than naive solution. Here we use sorted vector
  to store points left to the current. It allows us to quickly in O(log N)
  count number of points with Y-coord less than current, but nevertheless
  insertion in this vector can take O(N) in worst case. So worst case
  is still O(N*N) as in naive solution, but even so it passes all tests
  and the longest one takes 1.6s
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
    vector<double> y;
    for (int i = 0; i < n; i ++) {
        cin >> a[i].first;
        cin >> a[i].second;
    }

    sort(a.begin(), a.end(), [](pair<double, double> &x, pair<double, double> &y) {
      return x.first < y.first;
    });

    double nom = 0, denom = 0;

    for (int i = 0; i < n;) {
        int j = i;
        while (j < n && a[i].first == a[j].first) {
            auto l = lower_bound(y.begin(), y.end(), a[j].second) - y.begin();
            auto r = upper_bound(y.begin(), y.end(), a[j].second) - y.begin();
            nom += l + (r - l) / 2.0;
            denom += i;
            j ++;
        }

        for (int k = i; k < j; k ++) {
            auto l = lower_bound(y.begin(), y.end(), a[k].second);
            y.insert(l, a[k].second);
        }

        i = j;
    }

    cout << nom / denom;
}