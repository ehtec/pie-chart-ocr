#include <iostream>
#include <iomanip>

#include "color/color.hpp"

//compile with:
// g++ -I color/src -c -fPIC colorprocesser.cpp -o colorprocesser.o -O2
// g++ -shared -Wl,-soname,libcolorprocesser.so -o libcolorprocesser.so colorprocesser.o -O2


class ColorProcesser{

    public:
        const double helloworld(){
            return 0.123;
        }

        double test_calc(){

            ::color::rgb<double> a;  // = {1.0, 0.0, 0.0};
            ::color::rgb<double> b;  // = {0.0, 0.0, 1.0};

            a = ::color::rgb<double>({1.0, 0.0, 0.0});
            b = ::color::rgb<double>({0.0, 0.0, 1.0});

            double the_color_distance = ::color::operation::distance< ::color::constant::distance::CIEDE2000_entity >( a, b );

            return the_color_distance;

        }

        // RGB color distance. Not normalized, 0..255
        double color_distance(double r1, double g1, double b1, double r2, double g2, double b2){

            ::color::rgb<double> a({r1 / 255.0, g1 / 255.0, b1 / 255.0});
            ::color::rgb<double> b({r2 / 255.0, g2 / 255.0, b2 / 255.0});

            double the_color_distance = ::color::operation::distance< ::color::constant::distance::CIEDE2000_entity >( a, b );

            return the_color_distance;

        }

        // RGB color distance for array
        double* array_color_distance(double r1, double g1, double b1, double *r2, double *g2, double *b2, int m){

            ::color::rgb<double> a({r1 / 255.0, g1 / 255.0, b1 / 255.0});
            ::color::rgb<double> b;

            double *the_color_distances = new double[m];

            int i;

            for(i = 0; i < m; i++){

                b = ::color::rgb<double>({r2[i] / 255.0, g2[i] / 255.0, b2[i] / 255.0});

                the_color_distances[i] = ::color::operation::distance< ::color::constant::distance::CIEDE2000_entity >( a, b );

            }

            return the_color_distances;

        }

};


extern "C" {
    ColorProcesser* ColorProcesser_new(){ return new ColorProcesser; }
    void ColorProcesser_delete(ColorProcesser *colorprocesser){ delete colorprocesser; }
    const double ColorProcesser_helloworld(ColorProcesser* colorprocesser){ return colorprocesser->helloworld(); }
    double ColorProcesser_test_calc(ColorProcesser* colorprocesser){ return colorprocesser->test_calc(); }
    double ColorProcesser_color_distance(ColorProcesser* colorprocesser, double r1, double g1, double b1, double r2, double g2, double b2){

        return colorprocesser->color_distance(r1, g1, b1, r2, g2, b2);

    }

    double* ColorProcesser_array_color_distance(ColorProcesser* colorprocesser, double r1, double g1, double b1, double *r2, double *g2, double *b2, int m){

        return colorprocesser->array_color_distance(r1, g1, b1, r2, g2, b2, m);

    }

    void free_double_array(double* pointer){

        delete[] pointer;

    }
}



