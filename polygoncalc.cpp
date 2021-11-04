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

//#include "ndarray.h"

namespace bg = boost::geometry;

// use the following two commands to compile:
//
//WARNING: DO NOT USE O2. IT DESTROYS bg::correct FUNCTIONALITY
// g++ -lboost_system -c -fPIC polygoncalc.cpp -o polygoncalc.o -O2
// g++ -lboost_system -shared -Wl,-soname,libpolygoncalc.so -o libpolygoncalc.so polygoncalc.o -O2
// if it does not work, add -std=c++14 and add -I and -L directories

typedef boost::geometry::model::d2::point_xy<double> point_type;
//typedef bg::model::point<double, 2, bg::cs::cartesian> point_type;
typedef bg::model::polygon<point_type> polygon_type;

//typedef std::vector< std::vector<double>> Matrix;


void dump( const std::string & label, const std::list< std::set< int > > & values )
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


void combine( std::list< std::set< int > > & values )
{
    for( std::list< std::set< int > >::iterator iter = values.begin(); iter != values.end(); ++iter )
        for( std::list< std::set< int > >::iterator niter( iter ); ++niter != values.end(); )
            if( std::find_first_of( iter->begin(), iter->end(), niter->begin(), niter->end() ) != iter->end() )
            {
                iter->insert( niter->begin(), niter->end() );
                values.erase( niter );
                niter = iter;
            }
}


class PolygonCalc{

    public:
        const double helloworld(){
//            std::cout << "Hello World" << std::endl;
            return 0.123;
        }

        double test_calc(){

//            Point points[] = {Point(0,0), Point(1,0), Point(1,1), Point(0,1)};
//            Point points2[] = {Point(0.5,0.5), Point(1.5,0.5), Point(1.5, 1.5), Point(0.5,1.5)};

            polygon_type poly1 {{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}};
            polygon_type poly2 {{{5.5, 0.5}, {6.5, 0.5}, {6.5, 1.5}, {5.5, 1.5}}};

//            polygon_type poly1 {{{0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}}};
//            polygon_type poly2 {{{0.5, 1.5}, {1.5, 1.5}, {1.5, 0.5}, {0.5, 0.5}, {0.5, 1.5}}};

            bg::correct(poly1);
            bg::correct(poly2);

//            std::vector<point_type> points1;
//            std::vector<point_type> points2;
//
//            points1.push_back(point_type(0.0,0.0));
//            points1.push_back(point_type(1.0, 0.0));
//            points1.push_back(point_type(1.0, 1.0));
//            points1.push_back(point_type(0.0, 1.0));
//
//            points2.push_back(point_type(0.5, 0.5));
//            points2.push_back(point_type(1.5, 0.5));
//            points2.push_back(point_type(1.5, 1.5));
//            points2.push_back(point_type(0.5, 1.5));

//            boost::geometry::read_wkt(
//                "POLYGON((2 1.3,2.4 1.7,2.8 1.8,3.4 1.2,3.7 1.6,3.4 2,4.1 3,5.3 2.6,5.4 1.2,4.9 0.8,2.9 0.7,2 1.3)"
//                    "(4.0 2.0, 4.2 1.4, 4.8 1.9, 4.4 2.2, 4.0 2.0))", poly1);
//
//            boost::geometry::read_wkt(
//                "POLYGON((4.0 -0.5 , 3.5 1.0 , 2.0 1.5 , 3.5 2.0 , 4.0 3.5 , 4.5 2.0 , 6.0 1.5 , 4.5 1.0 , 4.0 -0.5))", poly2);

//            polygon_type poly1;
//            polygon_type poly2;
//
//            bg::assign_points(poly1, points1);
//            bg::assign_points(poly2, points2);

            std::deque<polygon_type> output;
            bg::intersection(poly1, poly2, output);

//            std::cout << bg::area(poly1) <<std::endl;
//            std::cout << bg::area(poly2) <<std::endl;


            double totalArea = 0.0;

            int i = 0;
            BOOST_FOREACH(polygon_type const& p, output)
            {

//                std::cout << i++ << ": " << boost::geometry::area(p) << std::endl;

                totalArea += bg::area(p);
            }

            double d = bg::distance(poly1, poly2);

            std::cout << d << std::endl;

            return totalArea;

        }

        int test_nparray(double *A, int n){

//        int M, N;
//
//        M = poly1.size();
//        N = poly1[0].size();
//
//        std::cout << M << std::endl << N << std::endl;

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
//                std::cout << poly1x[i] << " " << poly1y[i] << std::endl;
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

        double poly_intersection_area(double *poly1x, double *poly1y, double *poly2x, double *poly2y, int m, int n){

            int i;

            std::vector<point_type> points1;
            std::vector<point_type> points2;

            for (i=0; i<m; i++) {
                points1.push_back(point_type(poly1x[i], poly1y[i]));
//                std::cout << poly1x[i] << " " << poly1y[i] << std::endl;
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
//                std::cout << poly1x[i] << " " << poly1y[i] << std::endl;
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

        unsigned long* group_elements(unsigned long *a, unsigned long *b, unsigned long *c, unsigned long *d, unsigned long n)
        {
            
            unsigned long *element_groups = new unsigned long[n];
            
            std::vector<bool> v(n);
            std::fill(v.begin(), v.begin() + 2, true);
            
            do {
                
                for (unsigned long i = 0; i < n; i++) {
                    
                    if (v[i]) {
                        std::cout << (i + 1) << " ";
                    }
                                        
                }
                
                std::cout << std::endl;
                
            } while (std::prev_permutation(v.begin(), v.end()));
            
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
    double PolygonCalc_poly_intersection_area(PolygonCalc* polygoncalc, double *poly1x, double *poly1y, double *poly2x, double *poly2y, int m, int n){

        return polygoncalc->poly_intersection_area(poly1x, poly1y, poly2x, poly2y, m, n);

    }
    double PolygonCalc_poly_intersection_area_ratio(PolygonCalc* polygoncalc, double *poly1x, double *poly1y, double *poly2x, double *poly2y, int m, int n){

        return polygoncalc->poly_intersection_area_ratio(poly1x, poly1y, poly2x, poly2y, m, n);

    }
    
    unsigned long* PolygonCalc_group_elements(PolygonCalc* polygoncalc, unsigned long *a, unsigned long *b, unsigned long *c, unsigned long *d, unsigned long n) {
        
        return polygoncalc->group_elements(a, b, c, d, n);
        
    }
    
    void free_long_array(unsigned long* pointer){

        delete[] pointer;

    }
    
}

