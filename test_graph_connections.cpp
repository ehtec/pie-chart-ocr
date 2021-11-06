#include <bits/stdc++.h>

int merge(int* parent, int x)
{
    if (parent[x] == x)
        return x;
    return merge(parent, parent[x]);
}
int connectedcomponents(int n, std::vector<std::vector<int> >& edges)
{
    int parent[n];
    for (int i = 0; i < n; i++) {
        parent[i] = i;
    }
    for (auto x : edges) {
        parent[merge(parent, x[0])] = merge(parent, x[1]);
    }
    int ans = 0;
    for (int i = 0; i < n; i++) {
        ans += (parent[i] == i);
    }
    for (int i = 0; i < n; i++) {
        parent[i] = merge(parent, parent[i]);
    }
    std::map<int, std::list<int> > m;
    for (int i = 0; i < n; i++) {
        m[parent[i]].push_back(i);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        std::list<int> l = it->second;
        for (auto x : l) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
    return ans;
}
int main()
{
    int n = 5;
    std::vector<int> e1 = { 0, 1 };
    std::vector<int> e2 = { 2, 3 };
    std::vector<int> e3 = { 3, 4 };
    std::vector<std::vector<int> > e;
    e.push_back(e1);
    e.push_back(e2);
    e.push_back(e3);
    int a = connectedcomponents(n, e);
    std::cout << "total no. of connected components are: " << a
         << std::endl;
    return 0;
}

