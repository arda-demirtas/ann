#include <iostream>
#include "Layer.h"
#include <Eigen/Dense>
#include "Ann.h"

using namespace std;
using Eigen::MatrixXd;

int main()
{
    //xor gate   
	MatrixXd i(2, 4);
    Ann ann;
	i(0, 0) = 1;
	i(1, 0) = 1;
    i(0, 1) = 1;
    i(1, 1) = 0;
    i(0, 2) = 0;
    i(1, 2) = 1;
    i(0, 3) = 0;
    i(1, 3) = 0;
    MatrixXd o(1, 4);
    o(0, 0) = 0;
    o(0, 1) = 1;
    o(0, 2) = 1;
    o(0, 3) = 0;

    MatrixXd test1(2, 1);
    test1(0, 0) = 1;
    test1(1, 0) = 1;
    MatrixXd test2(2, 1);
    test2(0, 0) = 1;
    test2(1, 0) = 0;
    MatrixXd test3(2, 1);
    test3(0, 0) = 0;
    test3(1, 0) = 1;
    MatrixXd test4(2, 1);
    test4(0, 0) = 0;
    test4(1, 0) = 0;

    ann.addLayer(2, 2, "sigmoid");
    ann.addLayer(1, 2, "sigmoid");
    ann.learn(i, o, 10000);
    cout << ann.predict(test1) << endl;
    cout << ann.predict(test2) << endl;
    cout << ann.predict(test3) << endl;
    cout << ann.predict(test4) << endl;
	return 0;
}