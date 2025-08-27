#ifndef POINT_H
#define POINT_H

namespace mc_localizer {

class Point {
private:
    double x_, y_;

public:
    Point():
        x_(0.0), y_(0.0) {};

    Point(double x, double y):
        x_(x), y_(y) {};

    ~Point() {};

    inline void set_x(double x) { x_ = x; }
    inline void set_y(double y) { y_ = y; }
    inline void set_point(double x, double y) { x_ = x, y_ = y; }
    inline void set_point(Point p) { x_ = p.x_, y_ = p.y_; }

    inline double get_x(void) { return x_; }
    inline double get_y(void) { return y_; }
    inline Point get_point(void) { return Point(x_, y_); }

}; // class Point

} // namespace mc_localizer

#endif // POINT_H