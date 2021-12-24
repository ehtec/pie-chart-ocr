#include <bits/stdc++.h>

unsigned long merge(unsigned long* parent, unsigned long x)
{
    if (parent[x] == x)
        return x;
    return merge(parent, parent[x]);
}
unsigned long connectedcomponents(unsigned long n, std::vector<std::vector<unsigned long> >& edges)
{
    
    std::list<std::list<unsigned long>> res = {};
    
    unsigned long parent[n];
    for (unsigned long i = 0; i < n; i++) {
        parent[i] = i;
    }
    for (auto x : edges) {
        parent[merge(parent, x[0])] = merge(parent, x[1]);
    }
    unsigned long ans = 0;
    for (unsigned long i = 0; i < n; i++) {
        ans += (parent[i] == i);
    }
    for (unsigned long i = 0; i < n; i++) {
        parent[i] = merge(parent, parent[i]);
    }
    std::map<unsigned long, std::list<unsigned long> > m;
    for (unsigned long i = 0; i < n; i++) {
        m[parent[i]].push_back(i);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        std::list<unsigned long> l = it->second;

        for (auto x : l) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        
        res.push_back(l);
    }
    return ans;
}
int main()
{
    unsigned long n = 5;
    std::vector<unsigned long> e1 = { 0, 1 };
    std::vector<unsigned long> e2 = { 2, 3 };
    std::vector<unsigned long> e3 = { 3, 4 };
    std::vector<std::vector<unsigned long> > e;
    e.push_back(e1);
    e.push_back(e2);
    e.push_back(e3);
    unsigned long a = connectedcomponents(n, e);
    std::cout << "total no. of connected components are: " << a
         << std::endl;
    return 0;
}

