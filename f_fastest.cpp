/*
  This is the fastest solution that uses binary tree to store points left
  to the current. This allows us to add new point and query number
  of points less than current Y in O(log N) time. Thus whole time
  complexity is O(N * log N), and the longest test case takes only 126ms
*/

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

template <class T> struct Node {
    T x;
    int left_count = 0, right_count = 0, n = 1;
    Node *left = NULL;
    Node *right = NULL;

    Node(T x) : x(x) {}
};

template <class T> class BinaryTree {
    private:
        Node<T> *root = NULL;

    public:
        void add(T x) {
            if (this->root == NULL)
                this->root = new Node<T>(x);
            else {
                auto cur = this->root;
                while(1) {
                    if (x == cur->x) {
                        cur->n ++;
                        break;
                    } else if (x < cur->x) {
                        cur->left_count ++;
                        if (cur->left == NULL) {
                            cur->left = new Node<T>(x);
                            break;
                        } else
                            cur = cur->left;
                    } else {
                        cur->right_count ++;
                        if (cur->right == NULL) {
                            cur->right = new Node<T>(x);
                            break;
                        } else
                            cur = cur->right;
                    }
                }
            }
        }

        pair<int, int> count_less_and_eq(T x) {
            int less = 0, eq = 0;
            auto cur = this->root;
            while (cur) {
                if (x == cur-> x) {
                    less += cur->left_count;
                    eq = cur->n;
                    break;
                } else if (x < cur->x) {
                    cur = cur->left;
                } else {
                    less += cur->left_count + cur->n;
                    cur = cur->right;
                }
            }
            return make_pair(less, eq);
        }
};

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

    sort(a.begin(), a.end(), [](pair<double, double> &x, pair<double, double> &y) {
      return x.first < y.first;
    });

    BinaryTree<double> tree;
    double nom = 0, denom = 0;

    for (int i = 0; i < n;) {
        int j = i;
        while (j < n && a[i].first == a[j].first) {
            auto r = tree.count_less_and_eq(a[j].second);
            nom += r.first + r.second / 2.0;
            denom += i;
            j ++;
        }

        for (int k = i; k < j; k ++)
            tree.add(a[k].second);

        i = j;
    }

    cout << nom / denom;
}