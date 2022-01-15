#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include <set>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/assign.hpp>

#include <boost/foreach.hpp>

#include <stdlib.h>
#include <bits/stdc++.h>


namespace bg = boost::geometry;

// use the following two commands to compile:
//
//WARNING: DO NOT USE O2. IT DESTROYS bg::correct FUNCTIONALITY
// g++ -lboost_system -c -fPIC polygoncalc.cpp -o polygoncalc.o -O2
// g++ -lboost_system -shared -Wl,-soname,libpolygoncalc.so -o libpolygoncalc.so polygoncalc.o -O2
// if it does not work, add -std=c++14 and add -I and -L directories

typedef boost::geometry::model::d2::point_xy<double> point_type;

typedef bg::model::polygon<point_type> polygon_type;


void dump( const std::string & label, const std::list< std::set< unsigned long > > & values )
{
    std::cout << label << std::endl;
    for( auto iter : values )
    {
        std::cout << "{ ";
        for( auto val : iter )
            std::cout << val << ", ";
        std::cout << "}, ";
    }
    std::cout << std::endl;
}


void combine( std::list< std::set< unsigned long > > & values )
{
    for( std::list< std::set< unsigned long > >::iterator iter = values.begin(); iter != values.end(); ++iter )
        for( std::list< std::set< unsigned long > >::iterator niter( iter ); ++niter != values.end(); )
            if( std::find_first_of( iter->begin(), iter->end(), niter->begin(), niter->end() ) != iter->end() )
            {
                iter->insert( niter->begin(), niter->end() );
                values.erase( niter );
                niter = iter;
            }
}

unsigned long merge(unsigned long* parent, unsigned long x)
{
    if (parent[x] == x)
        return x;
    return merge(parent, parent[x]);
}

void connectedcomponents(unsigned long n, std::vector<std::vector<unsigned long> >& edges, std::list<std::list<unsigned long>>& res)
{
        
    unsigned long parent[n];
    for (unsigned long i = 0; i < n; i++) {
        parent[i] = i;
    }
    for (auto x : edges) {
        parent[merge(parent, x[0])] = merge(parent, x[1]);
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
        
        res.push_back(l);
    }
}

class PolygonCalc{

    public:
        const double helloworld(){
            return 0.123;
        }

        double test_calc(){

            polygon_type poly1 {{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}};
            polygon_type poly2 {{{5.5, 0.5}, {6.5, 0.5}, {6.5, 1.5}, {5.5, 1.5}}};

            bg::correct(poly1);
            bg::correct(poly2);

            std::deque<polygon_type> output;
            bg::intersection(poly1, poly2, output);

            double totalArea = 0.0;

            BOOST_FOREACH(polygon_type const& p, output)
            {

                totalArea += bg::area(p);
            }

            double d = bg::distance(poly1, poly2);

            std::cout << d << std::endl;

            return totalArea;

        }

        int test_nparray(double *A, int n){

            int i;
            double sum = 0.0;

            for (i=0; i<n; i++) {

                sum += A[i];

            }

            std::cout << "n: " << n << std::endl;

            std::cout << sum / n << std::endl;

            return n;

        }

        double min_poly_distance(double *poly1x, double *poly1y, double *poly2x, double *poly2y, int m, int n){

            int i;

            std::vector<point_type> points1;
            std::vector<point_type> points2;

            for (i=0; i<m; i++) {
                points1.push_back(point_type(poly1x[i], poly1y[i]));
            }

            for (i=0; i<n; i++) {
                points2.push_back(point_type(poly2x[i], poly2y[i]));
            }

            polygon_type poly1;
            polygon_type poly2;

            bg::assign_points(poly1, points1);
            bg::assign_points(poly2, points2);

            bg::correct(poly1);
            bg::correct(poly2);

            std::deque<polygon_type> output;
            bg::intersection(poly1, poly2, output);

            double totalArea = 0.0;

            BOOST_FOREACH(polygon_type const& p, output)
            {
                totalArea += bg::area(p);
            }

            if (totalArea > 0.0){
                return 0.0;
            }

            return bg::distance(poly1, poly2);

        }
        
        double poly_area(double *poly1x, double *poly1y, int m){

            int i;

            std::vector<point_type> points1;

            for (i=0; i<m; i++) {
                points1.push_back(point_type(poly1x[i], poly1y[i]));
            }

            polygon_type poly1;

            bg::assign_points(poly1, points1);

            bg::correct(poly1);

            double totalArea = bg::area(poly1);

            return totalArea;

        }

        double poly_intersection_area(double *poly1x, double *poly1y, double *poly2x, double *poly2y, int m, int n){

            int i;

            std::vector<point_type> points1;
            std::vector<point_type> points2;

            for (i=0; i<m; i++) {
                points1.push_back(point_type(poly1x[i], poly1y[i]));
            }

            for (i=0; i<n; i++) {
                points2.push_back(point_type(poly2x[i], poly2y[i]));
            }

            polygon_type poly1;
            polygon_type poly2;

            bg::assign_points(poly1, points1);
            bg::assign_points(poly2, points2);

            bg::correct(poly1);
            bg::correct(poly2);

            std::deque<polygon_type> output;
            bg::intersection(poly1, poly2, output);

            double totalArea = 0.0;

            BOOST_FOREACH(polygon_type const& p, output)
            {
                totalArea += bg::area(p);
            }

            return totalArea;

        }

        double poly_intersection_area_ratio(double *poly1x, double *poly1y, double *poly2x, double *poly2y, int m, int n){

            int i;

            std::vector<point_type> points1;
            std::vector<point_type> points2;

            for (i=0; i<m; i++) {
                points1.push_back(point_type(poly1x[i], poly1y[i]));
            }

            for (i=0; i<n; i++) {
                points2.push_back(point_type(poly2x[i], poly2y[i]));
            }

            polygon_type poly1;
            polygon_type poly2;

            bg::assign_points(poly1, points1);
            bg::assign_points(poly2, points2);

            bg::correct(poly1);
            bg::correct(poly2);

            std::deque<polygon_type> output;
            bg::intersection(poly1, poly2, output);

            double totalArea = 0.0;

            BOOST_FOREACH(polygon_type const& p, output)
            {
                totalArea += bg::area(p);
            }

            double area1 = bg::area(poly1);
            double area2 = bg::area(poly2);

            if (area1 > area2) {

                return totalArea / area2;

            } else {

                return totalArea / area1;

            }

            return totalArea;

        }

        unsigned long* old_group_elements(unsigned long *a, unsigned long *b,
            unsigned long *c, unsigned long *d, unsigned long n,
            double threshold_dist)
        {
            
            unsigned long *element_groups = new unsigned long[n];
            
            unsigned long i;
            
            std::vector<bool> v(n);
            std::fill(v.begin(), v.begin() + 2, true);
            
            std::vector<unsigned long> w(2);
            
            std::vector<point_type> points1;
            std::vector<point_type> points2;
            
            double height;
                                    
            double dist;
            
            double totalArea;
            
            std::vector< std::vector<unsigned long> > to_process = {};
            
            std::list< std::list<unsigned long> > res = {};

            
            std::cout << "threshold_dist: " << threshold_dist << std::endl;
            
            std::cout << "Computing polygon distances..." << std::endl;
            
            for (i = 0; i < n; i++) {
                
                to_process.push_back({i, i});
                
            }
            
            do {
                                
                w = {};
                
                points1 = {};
                points2 = {};
                                
                for (i = 0; i < n; i++) {
                    
                    if (v[i]) {
                        w.push_back(i);
                    }
                                        
                }

                points1.push_back(point_type(a[w[0]], b[w[0]]));
                points1.push_back(point_type(c[w[0]], b[w[0]]));
                points1.push_back(point_type(a[w[0]], d[w[0]]));
                points1.push_back(point_type(c[w[0]], d[w[0]]));
                
                points2.push_back(point_type(a[w[1]], b[w[1]]));
                points2.push_back(point_type(c[w[1]], b[w[1]]));
                points2.push_back(point_type(a[w[1]], d[w[1]]));
                points2.push_back(point_type(c[w[1]], d[w[1]]));
                                
                polygon_type poly1;
                polygon_type poly2;

                bg::assign_points(poly1, points1);
                bg::assign_points(poly2, points2);

                bg::correct(poly1);
                bg::correct(poly2);
                
                height = std::min(abs(double(d[w[0]] - b[w[0]])), abs(double(d[w[1]] - b[w[1]])));
                                
                std::deque<polygon_type> output;
                bg::intersection(poly1, poly2, output);

                totalArea = 0.0;

                BOOST_FOREACH(polygon_type const& p, output)
                {
                    totalArea += bg::area(p);
                }

                if (totalArea > 0.0){
                    dist = 0.0;
                } else {
                    dist = bg::distance(poly1, poly2);
                }
                                
                if (dist <= threshold_dist * height) {
                    to_process.push_back(w);
                }
                        
                
            } while (std::prev_permutation(v.begin(), v.end()));
                        
            std::cout << "Combining to nested list..." << std::endl;
            
            connectedcomponents(n, to_process, res);
            
            i = 0;
            
            for (auto const& el: res) {
                
                for (auto const& itm: el) {
                    element_groups[itm] = i;
                }
                
                i++;
                
            }
            
            return element_groups;
            
        }
        
        unsigned long* group_elements(double *a, double *b,
            double *c, double *d, unsigned long n,
            double threshold_dist, double slov_ratio, int size_metric_mode)
        {
            
            unsigned long *element_groups = new unsigned long[n];
            
            unsigned long i;
            
            std::vector<bool> v(n);
            std::fill(v.begin(), v.begin() + 2, true);
            
            std::vector<unsigned long> w(2);
            
            std::vector<point_type> points1;
            std::vector<point_type> points2;
            
            double height, width, size_metric, min_height;
            
            double min_x_dist, min_y_dist, min_normal_dist;
            
            double pre_threshold_dist = std::max(5.0, threshold_dist);
                                    
            double dist;
            
            double totalArea;
            
            double y1, y2;
            
            std::vector< std::vector<unsigned long> > to_process = {};
            
            std::list< std::list<unsigned long> > res = {};

            
//            std::cout << "threshold_dist: " << threshold_dist << std::endl;
            
//            std::cout << "Computing polygon distances..." << std::endl;
            
            for (i = 0; i < n; i++) {
                
                to_process.push_back({i, i});
                
            }
            
            do {
                                
                w = {};
                
                points1 = {};
                points2 = {};
                
                for (i = 0; i < n; i++) {
                    
                    if (v[i]) {
                        w.push_back(i);
                    }
                                        
                }
                
                height = std::max(abs(double(d[w[0]] - b[w[0]])), abs(double(d[w[1]] - b[w[1]])));
                
                min_height = std::min(abs(double(d[w[0]] - b[w[0]])), abs(double(d[w[1]] - b[w[1]])));
                
                width = std::max(abs(double(c[w[0]] - a[w[0]])), abs(double(c[w[1]] - a[w[1]])));
                
                switch (size_metric_mode) {
                    
                    case 0:
                        size_metric = std::max(height, width);
                        break;
                       
                    case 1:
                        size_metric = height;
                        break;
                        
                    case 2:
                        size_metric = width;
                        break;
                        
                    case 3:
                        size_metric = std::min(height, width);
                        break;
                        
                    default:
                        throw std::invalid_argument("Unknown size metric mode");
                    
                }
                
//                size_metric = std::max(height, width);
                
                min_x_dist = std::min({abs(double(a[w[0]] - c[w[1]])), abs(double(c[w[0]] - a[w[1]])), abs(double(a[w[0]] - a[w[1]])), abs(double(c[w[0]] - c[w[1]]))});
                
                min_y_dist = std::min({abs(double(b[w[0]] - d[w[1]])), abs(double(d[w[0]] - b[w[1]])), abs(double(b[w[0]] - b[w[1]])), abs(double(d[w[0]] - d[w[1]]))});
                
                min_normal_dist = std::min(min_x_dist, min_y_dist);
                
                if (min_normal_dist > pre_threshold_dist * size_metric) {
                    continue;
                }

                points1.push_back(point_type(a[w[0]], b[w[0]]));
                points1.push_back(point_type(c[w[0]], b[w[0]]));
                points1.push_back(point_type(a[w[0]], d[w[0]]));
                points1.push_back(point_type(c[w[0]], d[w[0]]));
                
                points2.push_back(point_type(a[w[1]], b[w[1]]));
                points2.push_back(point_type(c[w[1]], b[w[1]]));
                points2.push_back(point_type(a[w[1]], d[w[1]]));
                points2.push_back(point_type(c[w[1]], d[w[1]]));
                                
                polygon_type poly1;
                polygon_type poly2;

                bg::assign_points(poly1, points1);
                bg::assign_points(poly2, points2);

                bg::correct(poly1);
                bg::correct(poly2);
                                
                std::deque<polygon_type> output;
                bg::intersection(poly1, poly2, output);

                totalArea = 0.0;

                BOOST_FOREACH(polygon_type const& p, output)
                {
                    totalArea += bg::area(p);
                }

                if (totalArea > 0.0){
                    dist = 0.0;
                } else {
                    dist = bg::distance(poly1, poly2);
                }
                
                if (dist == 0) {
                    to_process.push_back(w);
                    continue;
                }
                
                if (b[w[0]] < d[w[0]]) {
                    y1 = b[w[0]];
                    y2 = d[w[0]];
                } else {
                    y1 = d[w[0]];
                    y2 = b[w[0]];
                }
                
                if (slov_ratio >= 0) {
                    if (! (((y1 <= b[w[1]] && b[w[1]] <= y2) && (y2 - b[w[1]]) / min_height >= slov_ratio ) || ((y1 <= d[w[1]] && d[w[1]] <= y2) && (d[w[1]] - y1) / min_height >= slov_ratio) || ((y1 <= b[w[1]]) && (y1 <= d[w[1]]) && (y2 >= b[w[1]]) && (y2 >= d[w[1]])))) {
                        continue;
                    }
                }
                                
                if (dist <= threshold_dist * size_metric) {
                    to_process.push_back(w);
                }
                        
                
            } while (std::prev_permutation(v.begin(), v.end()));
                        
//            std::cout << "Combining to nested list..." << std::endl;
            
            connectedcomponents(n, to_process, res);
            
            i = 0;
            
            for (auto const& el: res) {
                
                for (auto const& itm: el) {
                    element_groups[itm] = i;
                }
                
                i++;
                
            }
            
            return element_groups;
            
        }
        
};


extern "C" {
    PolygonCalc* PolygonCalc_new(){ return new PolygonCalc; }
    void PolygonCalc_delete(PolygonCalc *polygoncalc){ delete polygoncalc; }
    const double PolygonCalc_helloworld(PolygonCalc* polygoncalc){ return polygoncalc->helloworld(); }
    double PolygonCalc_test_calc(PolygonCalc* polygoncalc){ return polygoncalc->test_calc(); }
    int PolygonCalc_test_nparray(double *A, int n, PolygonCalc* polygoncalc){ return polygoncalc->test_nparray(A, n); }
    double PolygonCalc_min_poly_distance(PolygonCalc* polygoncalc, double *poly1x, double *poly1y, double *poly2x, double *poly2y, int m, int n){

        return polygoncalc->min_poly_distance(poly1x, poly1y, poly2x, poly2y, m, n);

    }
    double PolygonCalc_poly_area(PolygonCalc* polygoncalc, double *poly1x, double *poly1y, int m){ return polygoncalc->poly_area(poly1x, poly1y, m); }
    double PolygonCalc_poly_intersection_area(PolygonCalc* polygoncalc, double *poly1x, double *poly1y, double *poly2x, double *poly2y, int m, int n){

        return polygoncalc->poly_intersection_area(poly1x, poly1y, poly2x, poly2y, m, n);

    }
    double PolygonCalc_poly_intersection_area_ratio(PolygonCalc* polygoncalc, double *poly1x, double *poly1y, double *poly2x, double *poly2y, int m, int n){

        return polygoncalc->poly_intersection_area_ratio(poly1x, poly1y, poly2x, poly2y, m, n);

    }
    
    unsigned long* PolygonCalc_group_elements(PolygonCalc* polygoncalc, double *a, double *b, double *c, double *d, unsigned long n, double threshold_dist, double slov_ratio, int size_metric_mode) {
        
        return polygoncalc->group_elements(a, b, c, d, n, threshold_dist, slov_ratio, size_metric_mode);
        
    }
    
    void free_long_array(unsigned long* pointer){

        delete[] pointer;

    }
    
}

