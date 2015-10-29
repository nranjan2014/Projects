/***********************************************************************************************************************
 *
 * ECE 471 : Project 1 Augmentation part
 *
 *  testMpp.cpp - test routine to use MPP to process the synthetic dataset
 * Last Modified by/Date: Niloo Ranjan, 09/15/2015
 * Added the code for testing decision rule using not eqaul prior probability for the two classes
 * Also, modified the code to count if the class assigned was same as that in the testing data set 
 * for each sample
 *
 ***********************************************************************************************************************
 *
 * ECE 471 : Project 2 Augmentation part
 *
 * Last Modified by/Date: Niloo Ranjan, 10/07/2015
 * Addition: modified the command line argument option. Auugmented the original file with functions and code 
 * needed to perform the tasks for Project 2.
 * Added cases options to perform different task of project 2.
 * Case 1: Run: "Task 4.1: Use nX. Classify the test set using MAP with equal prior probability and test cases 1,2,3"
 * Case 2: Run: "Task 4.2: Use nX. Find a set of prior probability that would improve the classification accuracy." 
 * Case 3: Run: "Task 4.3: Use tX. Classify the test set using MAP with the prior probability you just found in Task 4.2." 
 * Case 4: Run: "Task 4.4: Use fX. Classify the test set using MAP with the prior probability you just found in Task 4.2."
 * Case 5: ROC Curve data points collection for nX data set
 * Case 6: ROC Curve data points collection for tX data set
 * Case 7: ROC Curve data points collection for fX data set
 *
 *************************************************************************************************************************
 *
 * ECE 471 : Project 3 Augmentation part
 *
 * Last Modified by/Date: Niloo Ranjan, 10/27/2015
 * Addition: Modified command line arguments options and augmented code to accomodate
 * the completion of tasks in Project 3
 * Case 1: kNN implementation using full Euclidean distance on data set nX
 * Case 2: kNN implementation using partial Euclidean distance on data set nX
 * Case 3: kNN implementation using full Euclidean distance on data set tX
 * Case 4: kNN implementation using partial Euclidean distance on data set tX
 * Case 5: kNN implementation using full Euclidean distance on data set fX
 * Case 6: kNN implementation using partial Euclidean distance on data set fX
 * Case 7: kNN implementation using full Minkowski distance with different p values on data set nX
 * Case 8: kNN implementation using partial Minkowski distance with different p values on data set nX
 * Case 9: kNN implementation using full Minkowski distance with different p values on data set tX
 * Case 10: kNN implementation using partial Minkowski distance with different p values on data set tX
 * Case 11: kNN implementation using full Minkowski distance with different p values on data set fX
 * Case 12: kNN implementation using partial Minkowski distance with different p values on data set fX
 * Case 13: classification using MAP with optimal prior probability found in Project 2 on data set nX
 * Case 14: classification using MAP with optimal prior probability found in Project 2 on data set tX
 * Case 15: classification using MAP with optimal prior probability found in Project 2 on data set fX
 * Case 16: 10 fold cross-validation on "fglass" data set with kNN as the classifier
 *
 ************************************************************************************************************************/

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include "Matrix.h"             
#include "Pr.h"
#include <ctime>

using namespace std;
//#define CLOCK_PER_SEC 60
#define Usage "Usage: ./testMpp training_set test_set classes features cases \n\t training_set: the file name for training set\n\t test_set: the file name for test set\n\t classes: number of classes\n\t features: number of features (dimension)\n\t cases: used to run diffetent task categories, cases can be from 1 to 16\n\t K-values: for kNN \n\t p-values: for degrees of Minkowski distancei\n\n"

int main(int argc, char **argv)
{
    int nrTr, nrTe;       // number of rows in the training and test set
    int  nc;              // number of columns in the data set; both the 
    // training and test set should have the same
    // column number
    Matrix XTr, XTe;      // original training and testing set
    Matrix Tr, Te;        // training and testing set file received from command line

    // check to see if the number of argument is correct
    if (argc < 7) {
        cout << Usage;
        exit(1);
    }

    int classes = atoi(argv[3]);   // number of classes
    int nf = atoi(argv[4]);        // number of features (dimension)
    int cases = atoi(argv[5]);     // number of features (dimension)

    int K = atoi(argv[6]);
    double minK = atof(argv[7]);
    // read in data from the data file
    nc = nf+1; // the data dimension; plus the one label column

    XTr = readData(argv[1], nc);
    nrTr = XTr.getRow();          // get the number of rows in training set
    XTe = readData(argv[2], nc);
    nrTe = XTe.getRow();           // get the number of rows in testing set

    Matrix X;

    X = subMatrix(XTr, 0,0, nrTr-1, nf-1);

    // prepare the labels and error count
    Matrix labelMPP(nrTe, 1);      // a col vector to hold result for MPP
    int CorrectCountMPP = 0;       // calcualte error rate 

    Matrix means, covr, covarxN, sigma, nXTr, nXTe, D_M, V_M, B_M, E_M  ;
    // B_M basis vector
    // E_M eigen vector corresponding to eigen values error rate less than 0.10

    // calculate the mean and the variance of the original data set as whole without class
    // consideration
    sigma.createMatrix(nf, 1);
    means = mean(X,nf);
    covr = cov(X, nf);

    nXTr = XTr;
    // normalize the traning data set with sigma from covariance and mean already calculated
    for (int j = 0; j < nf ; j++)
    {
        sigma(j,0) = sqrt(covr(j,j));
        for (int i =0; i < nrTr; i++)
        {
            nXTr(i,j) = (nXTr(i,j) -  means(j,0)) / sigma(j,0);

        }
    }
    nXTe = XTe;
    // normalize the testing data set with sigma from covariance and mean already calculated
    for (int j = 0; j < nf ; j++)
    {
        sigma(j,0) = sqrt(covr(j,j));
        for (int i =0; i < nrTe; i++)
        {

            nXTe(i,j) = (nXTe(i,j) - means(j,0)) / sigma(j,0);
        }
    }

    // perform the transformation of normalized data set using PCA
    Matrix tXTr;
    Matrix tXTe;

    // get optimal basis vector
    B_M = GetPCA(nXTr);
    // using this basis vector to transformed training and testing data set
    tXTr = GetDataTransformedFLD ( nXTr, B_M);
    tXTe = GetDataTransformedFLD ( nXTe, B_M);


    // perform the transformation of normalized data set using FLD
    Matrix W, fXTr, fXTe;
    // get the optimal projection direction W
    W = GetFLD(nXTr);

    // using this optimal projection direction to transformed training and testing data set
    fXTr = GetDataTransformedFLD ( nXTr,W);
    fXTe = GetDataTransformedFLD ( nXTe,W);

    // kNN basic implementation using full Euclidean distance on data set nX
    if ( cases == 1)
    {
        Matrix label (nXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierEuclidian() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with full ecludian distance
        clock_t start = clock();

        for (int i = 0; i < nXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(nXTe, i,0,i, nXTe.getCol()-1);

            label(i,0) = KNNClassifierEuclidian(nXTr, sample, K);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, nXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
    }
    // kNN implementation using partial Euclidean distance on data set nX
    else if ( cases == 2)
    {
        Matrix label (nXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierPartialEuclidian() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with partial ecludian distance
        clock_t start = clock();

        for (int i = 0; i < nXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(nXTe, i,0,i, nXTe.getCol()-1);

            label(i,0) = KNNClassifierPartialEuclidian(nXTr, sample, K);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, nXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;

    }
    // kNN implementation using full Euclidean distance on data set tX
    else if ( cases == 3)
    {
        Matrix label (tXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierEuclidian() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with full ecludian distance
        clock_t start = clock();

        for (int i = 0; i < tXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(tXTe, i,0,i, tXTe.getCol()-1);

            label(i,0) = KNNClassifierEuclidian(tXTr, sample, K);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, tXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;

    }
    // kNN implementation using partial Euclidean distance on data set tX
    else if ( cases == 4)
    {
        Matrix label (tXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierPartialEuclidian() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with partial ecludian distance
        clock_t start = clock();

        for (int i = 0; i < tXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(tXTe, i,0,i, tXTe.getCol()-1);

            label(i,0) = KNNClassifierPartialEuclidian(tXTr, sample, K);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, tXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;

    }
    // kNN implementation using full Euclidean distance on data set fX
    else if ( cases == 5)
    {
        Matrix label (fXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierEuclidian() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with full ecludian distance
        clock_t start = clock();

        for (int i = 0; i < fXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(fXTe, i,0,i, fXTe.getCol()-1);

            label(i,0) = KNNClassifierEuclidian(fXTr, sample, K);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, fXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;

    }
    // kNN implementation using partial Euclidean distance on data set fX
    else if ( cases == 6)
    {
        Matrix label (fXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierPartialEuclidian() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with partial ecludian distance
        clock_t start = clock();

        for (int i = 0; i < fXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(fXTe, i,0,i, fXTe.getCol()-1);

            label(i,0) = KNNClassifierPartialEuclidian(fXTr, sample, K);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, fXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;

    }
    // kNN implementation using full Minkowski distance on data set nX
    else if ( cases == 7)
    {
        Matrix label (nXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierMinkowski() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with Minkowski distance
        clock_t start = clock();

        for (int i = 0; i < nXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(nXTe, i,0,i, nXTe.getCol()-1);

            label(i,0) = KNNClassifierMinkowski(nXTr, sample, K,minK);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, nXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;

    }
    // kNN implementation using partial Minkowski distance on data set nX
    else if ( cases == 8)
    {
        Matrix label (nXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierPartialMinkowski() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with Minkowski distance
        clock_t start = clock();

        for (int i = 0; i < nXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(nXTe, i,0,i, nXTe.getCol()-1);

            label(i,0) = KNNClassifierPartialMinkowski(nXTr, sample, K,minK);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, nXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;

    }
    // kNN implementation using full Minkowski distance on data set tX
    else if ( cases == 9)
    {
        Matrix label (tXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierMinkowski() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with Minkowski distance
        clock_t start = clock();

        for (int i = 0; i < tXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(tXTe, i,0,i, tXTe.getCol()-1);

            label(i,0) = KNNClassifierMinkowski(tXTr, sample, K,minK);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, tXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;

    }
    // kNN implementation using partial Minkowski distance on data set tX
    else if ( cases == 10)
    {
        Matrix label (tXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierPartialMinkowski() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with Minkowski distance
        clock_t start = clock();

        for (int i = 0; i < tXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(tXTe, i,0,i, tXTe.getCol()-1);

            label(i,0) = KNNClassifierPartialMinkowski(tXTr, sample, K,minK);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, tXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;

    }
    // kNN implementation using full Minkowski distance on data set fX
    else if ( cases == 11)
    {
        Matrix label (fXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierMinkowski() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with Minkowski distance
        clock_t start = clock();

        for (int i = 0; i < fXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(fXTe, i,0,i, fXTe.getCol()-1);

            label(i,0) = KNNClassifierMinkowski(fXTr, sample, K,minK);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, fXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
    }
    // kNN implementation using partial Minkowski distance on data set fX
    else if ( cases == 12)
    {
        Matrix label (fXTe.getRow(), 1);

        // for each sample in testing set call the KNNClassifierPartialMinkowski() to determine the class of the testing sample
        // Also time the function execution time for classification using KNN with Minkowski distance
        clock_t start = clock();

        for (int i = 0; i < fXTe.getRow(); i++)
        {
            Matrix sample = subMatrix(fXTe, i,0,i, fXTe.getCol()-1);

            label(i,0) = KNNClassifierPartialMinkowski(fXTr, sample, K,minK);
        }
        clock_t end = clock();

        DerivePerferformanceMetric ( label, fXTe, cases);

        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
    }
    // perform the task of classification using MAP on data set nX with optimal Prior Probability found in project 2
    else if ( cases == 13)
    {
        // using optimal prior probability found from task before
        // which is Pw1 = 0.7 and Pw2 = 0.3
        Matrix Pw2(2, 1);
        Pw2(0,0) = 0.7;
        Pw2(1,0) = 0.3;

        // get the computational time to do the classification
        clock_t start = clock();
        ClassificationWithBestPW ( nXTr, nXTe,Pw2, cases );
        clock_t end = clock();
        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
    }
    // perform the task of classification using MAP on data set tX with optimal Prior Probability found in project 2
    else if ( cases == 14)
    {
        // using optimal prior probability found in project 2
        // which is Pw1 = 0.7 and Pw2 = 0.3
        Matrix Pw2(2, 1);
        Pw2(0,0) = 0.7;
        Pw2(1,0) = 0.3;

        // get the computational time to do the classification
        clock_t start = clock();
        ClassificationWithBestPW ( tXTr, tXTe,Pw2, cases );
        clock_t end = clock();
        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
    }
    // perform the task of classification using MAP on data set fX with optimal Prior Probability found in project 2
    else if ( cases == 15)
    {
        // using optimal prior probability found in project 2
        // which is Pw1 = 0.7 and Pw2 = 0.3
        Matrix Pw2(2, 1);
        Pw2(0,0) = 0.7;
        Pw2(1,0) = 0.3;

        // get the computational time to do the classification
        clock_t start = clock();
        ClassificationWithBestPW ( fXTr, fXTe, Pw2, cases );
        clock_t end = clock();
        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
    }
    // 10 fold cross-validation on "fglass" data set with kNN as the classifier with full Euclidean distance
    else if (cases == 16)
    {
        int fea = 9;   // there are 9 features in this data set
        int sumError = 0;
        int sumTotalCount = 0;
        double totalTime = 0.0;

        //read in the flassdata file in a matrix
        Matrix glassData = readData("fglass.dat", fea+1);

        int Grow = 0;   // number of sample in this data set

        Grow = glassData.getRow();

        Matrix S, M, C;  // S: standard deviation, M: mean of whole data set, C: covariance of whole data set

        S.createMatrix(fea, 1);

        M = mean(glassData,fea);

        C = cov(glassData, fea);

        // normalize the fglass data set with sigma from covariance and mean already calculated above
        for (int j = 0; j < fea ; j++)
        {
            S(j,0) = sqrt(C(j,j));
            for (int i =0; i < Grow; i++)
            {
                glassData(i,j) = (glassData(i,j) - M(j,0)) / S(j,0);

            }
        }

        // max number of sample in the fold
        // this will be used to make the fold matrix with same number of 
        // column size
        int sampFold = 28;  

        // read in fold data
        Matrix foldData = readData("fglass.grp", sampFold );

        // read one row at a time from the 10 folds and assign the current row read as 
        // fold for building validating set. Then assign rest of the current to use as 
        // training set using kNN classifier c;assify the validating set from training set
        for (int i = 0; i < foldData.getRow(); i++ )
        {
            // current fold 
            Matrix S = subMatrix(foldData, i,0,i,foldData.getCol()-1);

            // build the validating set from the one fold just read
            Matrix test = getTestingData(S, glassData);

            // build the training set from leaving the current fold and including the rest of the
            // fold 
            Matrix training = getTrainingData(foldData, glassData, i );

            // to hold class label for testing samples
            Matrix label (test.getRow(), 1);


            int foldclass = 7; // there are 7 class category in this data set
            int ErrorCount = 0;

            // start timming the kNN classification completion for one fold 
            // of the validating set
            clock_t start = clock();

            // for each sample in the validating set use the kNN  
            // with original Euclidean disable to find the class label of the sample
            for (int i = 0; i < test.getRow(); i++)
            {
                Matrix sample = subMatrix(test, i,0,i, test.getCol()-1);

                label(i,0) = KNNClassifierEuclidianFold(training, sample, K, foldclass);

                if (label(i,0) != test(i,test.getCol()-1))
                {
                    ErrorCount++;
                }
            }

            sumError = sumError + ErrorCount;
            sumTotalCount = sumTotalCount + test.getRow();

            cout << "Error Rate: " << (((float) ErrorCount) / test.getRow())*100 << endl;

            clock_t end = clock();
            totalTime = totalTime + (((double) (end-start)) / 1000000);
            cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
        }
        cout << "Average Error Rate: " << (((float) sumError) / sumTotalCount)*100 << endl;
        cout << "Average Running Time: " << totalTime / 10 << " seconds" << endl;
    }
    else
    {
        cout << "Please specify the case from 1 to 16, which task to run" << endl;
    }

    return 0;
}

// this function classify the data set passed using optimal prior probability found
void ClassificationWithBestPW ( Matrix &trn, Matrix &tet, Matrix &BestPW, int type )
{
    int row;
    int col;
    row = tet.getRow();
    col = tet.getCol();
    Matrix labelMPP(row, 1);  // a col vector to hold result for MAP
    Matrix temp;
    temp = tet;

    for (int i=0; i<row; i++) {
        // classify one test sample at a time, get one sample from the test data
        Matrix sample = transpose(subMatrix(temp, i, 0, i, col-2));

        // call MPP to perform classification
        labelMPP(i,0) = mpp(trn, sample, 2 , 4, BestPW);
    }

    DerivePerferformanceMetric ( labelMPP, tet,type );

}

// this function calculates the performance metrics for the classification rule MAP used
// on the each type of data set tested
void DerivePerferformanceMetric ( Matrix & tested, Matrix &truth, int datatype)
{
    double Accuracy;
    int TP; // true positive number
    int TN; // true negative number
    int FP; // false positive number
    int FN; // false negative number
    TN = 0; 
    TP = 0; 
    FP = 0; 
    FN = 0; 

    Accuracy = 0.0;

    int row = truth.getRow();
    int col = truth.getCol();

    for (int i=0; i<row; i++) {
        if (tested(i,0) == truth(i, col-1)) 
        {
            if ( truth(i,col-1) == 1) // truth yes, observed yes
            {
                TP++;
            }
            else    // truth no, observed no
            {
                TN++;
            }
        }
        else
        {
            if (truth(i,col-1) == 1) // truth yes, observed no
            {
                FN++;
            }
            else    // truth no, observed yes
            {
                FP++;
            }
        }
    }

    Accuracy = ((double)((TP+TN)))/(TP+TN+FN+FP);
    cout << "The Accuracy rate using " << datatype << " is " << ((float)Accuracy) * 100 << endl;
}

// this function finds the optimal projection direction for FLD
Matrix GetFLD ( Matrix &train)
{
    static int nf;

    static Matrix *means, *covs;

    int nctr, nrtr;

    int i, j, n, c;

    Matrix SW, invSW, ret, tmp;

    nctr = train.getCol();

    nrtr = train.getRow();

    nf = nctr-1;

    c = 2;

    means = (Matrix *) new Matrix [c];

    for (i=0; i<c; i++)
        means[i].createMatrix(nf, 1);

    covs = (Matrix *) new Matrix [c];

    for (i=0; i<c; i++)
        covs[i].createMatrix(nf, nf);

    // get means and covariance of the classes
    for (i=0; i<c; i++) {
        tmp = getType(train, i);
        n = tmp.getRow();
        // get sw1 and sw2 sw = (n-1) covariance
        covs[i] = (n-1) * cov(tmp, nf);
        // find m1 and m2
        means[i] = mean(tmp, nf);
    }
    SW.createMatrix(nf, nf);

    // get sw = sw1 + sw2
    for (i=0; i < c; i++) {
        SW = SW + covs[i];
    }

    invSW = inverse(SW);

    // find W = sw+inverse + ( m1 - m2)
    ret = invSW->*(means[0] - means[1]);

    return ret;
}
// this function transforms the normalized training and the testing set
// W = optimal projection direction for FLD case
// W = basis vector for PCA case
Matrix GetDataTransformedFLD ( Matrix &nX, Matrix W)
{
    Matrix temp3, temp4, fx, fxte;
    int row;
    int col;

    row = nX.getRow();
    col = nX.getCol();

    temp3 = subMatrix(nX, 0, 0,row-1, col-2 );

    // for FLD it is y = W_transpose * X
    // for PCA it is y = B_transpose * X
    temp3 = transpose(W) ->* transpose( temp3);
    temp3 = transpose(temp3);

    fx.createMatrix(row, temp3.getCol()+1);

    // add the class label to the transformed data set
    for ( int i =0; i < row; i++ )
    {
        int j;
        for ( j =0; j < temp3.getCol(); j++)
        {
            fx(i,j) = temp3(i,j);
        }
        fx (i, j) = nX(i, col-1);
    }

    return fx;
}

// this function calculets the eigen values, eigen vector, basis vector for PCA 
Matrix GetPCA( Matrix &nX)
{

    Matrix temp, D_M, V_M, B_M, E_M, covarxN ;
    temp = nX;
    int row;
    int col;

    row = nX.getRow();
    col = nX.getCol();


    D_M.createMatrix(1,col-1);
    V_M.createMatrix(col-1,col-1);
    covarxN.createMatrix(col-1, col-1);

    temp = subMatrix(nX, 0,0, row-1, col-2);
    covarxN = cov(temp, col-1);

    // find the eigen values and eigen vector from the covariance matrix
    // of the normalized data set
    jacobi(covarxN, D_M, V_M );

    // sort the eigen values and rearrange the eigen vector accordingly
    eigsrt(D_M, V_M);

    double Sum_D_M = 0;
    double D_M_add = 0;
    int count =0;
    int eigncol = V_M.getCol();
    int eignrow = V_M.getRow();
    int i, j;

    // get the sum of all eigen values
    for ( i =0; i < eigncol; i++)
    {
        Sum_D_M += D_M(0,i);
    }

    // keep the eigen values that will give eoor rate less than 0.10
    for (  j = 0; j < eigncol; j++)
    {
        D_M_add += D_M(0,j);

        if((D_M_add / Sum_D_M) > 0.10)
        {
            break;
        }
    }

    // basic vector with higher values of eigen values
    B_M = subMatrix(V_M, 0, j, eignrow-1, eigncol-1);
    E_M = subMatrix(D_M, 0, j, 0, eigncol-1);

    return B_M;
}

// this function implement the basic implementattion of the kNN using 
// original Euclidean distance to be used to clasiify the testing set of 
// nX, tX, fX data set which is normalized, PCA transformed, FLD trasformed respectively	
int KNNClassifierEuclidian(Matrix &nXTr, Matrix &sample, int K)
{
    Matrix distance(nXTr.getRow(),2);
    Matrix samp;
    double sumDistance = 0.0;

    samp = subMatrix(sample, 0,0,0,sample.getCol()-1);

    // calculate the original Euclidean distance between the current tesing sample and each of the 
    // sample in the training set
    for ( int i =0; i < nXTr.getRow(); i++)
    {
        sumDistance = 0.0;
        Matrix samp1 = subMatrix(nXTr, i, 0,i,nXTr.getCol()-2);

        // get the square of the difference distance for each feature of the 
        for ( int j =0; j < samp1.getCol(); j++)
        {
            sumDistance = sumDistance + pow((samp(0,j) - samp1(0,j)), 2);
        }

        sumDistance = sqrt(sumDistance);

        // keep the disance calculated from each training sample along 
        // with their class label info
        distance(i,0) = sumDistance;
        distance(i,1) = nXTr(i, nXTr.getCol()-1);
    }

    // get just the distance part as row to use insersort function to sort the distances
    Matrix DisUnSort = transpose(subMatrix(distance, 0, 0, distance.getRow()-1, 0));

    Matrix DisSort(DisUnSort.getRow(), DisUnSort.getCol()); // this will keep sorted distance
    Matrix DisPos(DisUnSort.getRow(), DisUnSort.getCol());  // this will keep the index 
    // of the sorted distance 

    insertsort(DisUnSort, DisSort, DisPos);
    Matrix Kdistance(K, 2); // to keep only K nearest distances

    // get only the k nearest distance from the list of all calcualted sorted 
    // distance along with the label for the sample corresponding at that distance apart
    for ( int i =0; i< K; i++)
    {
        int pos = DisPos(0,i);
        int label = distance(pos,1);
        Kdistance(i,0) = DisSort (0,i);
        Kdistance(i,1) = label;
    }

    int maxNum = 0;
    int Class = 0;

    // as in this case there are only two class category 0 or 1 in the 
    // get the number of sample fall withing that K nearest neighborhood 
    // for each type and assign the class corresponding to the class with 
    // max number of sample found 
    Matrix Type0 = getType(Kdistance,0);
    Matrix Type1 = getType(Kdistance,1);
    if (Type0.getRow() > Type1.getRow())
    {
        Class= 0;
    }
    else
    {
        Class = 1;
    }
    return(Class);
}


// this function implement the kNN using partial Euclidean distance 
// to be used to clasiify the testing set of 
// nX, tX, fX data set which is normalized, PCA transformed, FLD trasformed respectively
int KNNClassifierPartialEuclidian(Matrix &nXTr, Matrix &sample, int K)
{
    Matrix distance(K,2);
    Matrix samp;
    double dis = 0.0;
    int D = 0;
    Matrix Kdistance(K, 2);

    samp = subMatrix(sample, 0,0,0,sample.getCol()-1);

    // get the original Euclidean distance from the first K sample only
    // with its class label information
    for ( int i =0; i < K; i++)
    {
        Matrix samp1 = subMatrix(nXTr, i, 0,i,nXTr.getCol()-2);

        dis = calculatePartDis(samp,samp1);

        distance(i,0) = dis;
        distance(i,1) = nXTr(i, nXTr.getCol()-1);
    }

    // sort the computed K distances
    Kdistance  = ComputeSortedDis( distance);

    double maxDis = Kdistance(K-1,0);
    D = 0; // untill what dimention to calculate the partial distance

    // for all K+1 onwards sample in the traning set 
    // compute the partial Euclidean distance by adding the distance between 
    // each feature one at a time and checking if adding the next feature distance would 
    // pass the calculated maximum disstance in the K kept sorted distance. 

    for ( int i = K; i < nXTr.getRow(); i++)
    {
        dis = 0;

        Matrix s1 = subMatrix(nXTr,i,0,i,D);
        Matrix s2 = subMatrix(sample,0,0,0,D);

        dis =  calculatePartDis(s1,s2);

        // check if the partial distance is still less than max distance found
        // also check not surpassing the total dimention 
        while(D < (nXTr.getCol()-2 )|| dis < maxDis )
        {
            // that means partial distance is still less 
            // than max distance and the boundary of the dimention has not reached
            if (D != nXTr.getCol()-1 )
            {
                // get the feature vector from both testing and training sample 
                // untill D dimention only
                s1 = subMatrix(nXTr, i,0,i,D);

                s2 = subMatrix(sample,0,0,0,D);

                // get the calculated partial distance
                dis = calculatePartDis(s1,s2);
                D++;
            }
            else
            {
                break;
            }
        }
        // this means calculated distance all the way to whole dimention 
        // and the calculated distance is less than the max distance kept in 
        // K sorted distance so far. 
        if ( D == nXTr.getCol()-1 && dis < maxDis  )
        {
            // include that distance in the 
            // K nearest distance list and remove the current max from the list
            // also sort the distance list in increasing order again so that max 
            // distance is at the end of the list
            Kdistance(K-1,0) = dis;
            Kdistance(K-1,1) = nXTr(i,nXTr.getCol()-1);
            Kdistance = ComputeSortedDis(Kdistance);

            // get the current max from the kept K nearest distance so far list
            maxDis = Kdistance(K-1,0);
            D = 0;
        }
    }

    // after going through the finding of K nearest neighbor
    // get the number of sample fall withing that K nearest neighborhood 
    // for each type and assign the class corresponding to the class with 
    // max number of sample found 
    int maxNum = 0;
    int Class = 0;
    Matrix Type0 = getType(Kdistance,0);
    Matrix Type1 = getType(Kdistance,1);
    if (Type0.getRow() > Type1.getRow())
    {
        Class= 0;
    }
    else
    {
        Class = 1;
    }
    return(Class);
}

// this function calculates the partial Euclidean distance to be used for 
// kNN implementation using Partial Euclidian distance
double calculatePartDis(Matrix &s1, Matrix &s2)
{
    double d = 0;
    int col = s1.getCol();

    for ( int i =0; i < col; i++)
    {
        d = d + pow((s1(0,i) - s2(0,i)), 2);
    }
    d = sqrt(d);

    return d;
}

// this method gets the unsorted computed K distance 
// and sorts the K distances using insersort() method from 
// provided matrix library
Matrix ComputeSortedDis( Matrix &UnSortDis)
{
    // get just the distance part as row to use insersort function to sort the distances
    Matrix DisUnSort = transpose(subMatrix(UnSortDis, 0, 0, UnSortDis.getRow()-1, 0));

    Matrix DisSort(DisUnSort.getRow(), DisUnSort.getCol()); // this will keep sorted distance
    Matrix DisPos(DisUnSort.getRow(), DisUnSort.getCol());  // this will keep the index 
    // of the sorted distance 

    insertsort(DisUnSort, DisSort, DisPos);
    Matrix KdistanceSorted(DisSort.getCol(), 2);  // to keep only K nearest distances

    // get only the k nearest distance from the list of all calcualted sorted 
    // distance along with the label for the sample corresponding at that distance apart
    for ( int i =0; i < DisSort.getCol(); i++)
    {
        int pos = DisPos(0,i);
        int label = UnSortDis(pos,1);
        KdistanceSorted(i,0) = DisSort (0,i);
        KdistanceSorted(i,1) = label;
    }
    return KdistanceSorted;
}

// this function implement the kNN using original Minkowski distance 
// to be used to clasiify the testing set of 
// nX, tX, fX data set which is normalized, PCA transformed, FLD trasformed respectively
int KNNClassifierMinkowski(Matrix &nXTr, Matrix &sample, int K, double MinP)
{
    Matrix distance(nXTr.getRow(),2);
    Matrix samp;
    double sumDistance = 0.0;

    samp = subMatrix(sample, 0,0,0,sample.getCol()-1);

    // calculate the original Minkowski distance using provided P value (MinP)between the current tesing sample and each of the 
    // sample in the training set 
    for ( int i =0; i < nXTr.getRow(); i++)
    {
        sumDistance = 0.0;
        Matrix samp1 = subMatrix(nXTr, i, 0,i,nXTr.getCol()-2);

        // get the square of the difference distance for each feature of the
        for ( int j =0; j < samp1.getCol(); j++)
        {
            sumDistance = sumDistance + pow(fabs((samp(0,j) - samp1(0,j))), MinP);
        }

        sumDistance = pow(sumDistance, 1/MinP);

        // keep the disance calculated from each training sample along 
        // with their class label info
        distance(i,0) = sumDistance;
        distance(i,1) = nXTr(i, nXTr.getCol()-1);
    }

    // get just the distance part as row to use insersort function to sort the distances
    Matrix DisUnSort = transpose(subMatrix(distance, 0, 0, distance.getRow()-1, 0));

    Matrix DisSort(DisUnSort.getRow(), DisUnSort.getCol()); // this will keep sorted distance
    Matrix DisPos(DisUnSort.getRow(), DisUnSort.getCol());  // this will keep the index 
    // of the sorted distance

    insertsort(DisUnSort, DisSort, DisPos);
    Matrix Kdistance(K, 2);   // to keep only K nearest distances

    // get only the k nearest distance from the list of all calcualted sorted 
    // distance along with the label for the sample corresponding at that distance apart
    for ( int i =0; i< K; i++)
    {
        int pos = DisPos(0,i);
        int label = distance(pos,1);
        Kdistance(i,0) = DisSort (0,i);
        Kdistance(i,1) = label;
    }

    int maxNum = 0;
    int Class = 0;

    // as in this case there are only two class category 0 or 1 in the 
    // get the number of sample fall withing that K nearest neighborhood 
    // for each type and assign the class corresponding to the class with 
    // max number of sample found 
    Matrix Type0 = getType(Kdistance,0);
    Matrix Type1 = getType(Kdistance,1);
    if (Type0.getRow() > Type1.getRow())
    {
        Class= 0;
    }
    else
    {
        Class = 1;
    }
    return(Class);
}

// this function implement the kNN using partial Minkowski distance 
// to be used to clasiify the testing set of 
// nX, tX, fX data set which is normalized, PCA transformed, FLD trasformed respectively
int KNNClassifierPartialMinkowski(Matrix &nXTr, Matrix &sample, int K, double minK)
{
    Matrix distance(K,2);
    Matrix samp;
    double dis = 0.0;
    int D = 0;
    Matrix Kdistance(K, 2);

    samp = subMatrix(sample, 0,0,0,sample.getCol()-1);

    // get the original Minkowski distance from the first K sample only
    // with its class label information
    for ( int i =0; i < K; i++)
    {
        Matrix samp1 = subMatrix(nXTr, i, 0,i,nXTr.getCol()-2);

        dis = calculatePartDisMinkowski(samp,samp1, minK);

        distance(i,0) = dis;
        distance(i,1) = nXTr(i, nXTr.getCol()-1);
    }

    // sort the computed K distances
    Kdistance  = ComputeSortedDis( distance);

    double maxDis = Kdistance(K-1,0);
    D = 0;  // untill what dimention to calculate the partial distance

    // for all K+1 onwards sample in the traning set 
    // compute the partial Euclidean distance by adding the distance between 
    // each feature one at a time and checking if adding the next feature distance would 
    // pass the calculated maximum disstance in the K kept sorted distance. 
    for ( int i = K; i < nXTr.getRow(); i++)
    {

        dis = 0;

        Matrix s1 = subMatrix(nXTr,i,0,i,D);
        Matrix s2 = subMatrix(sample,0,0,0,D);

        dis =  calculatePartDisMinkowski(s1,s2,minK );

        // check if the partial distance is still less than max distance found
        // also check not surpassing the total dimention
        while(D < (nXTr.getCol()-2 )|| dis < maxDis )
        {
            // that means partial distance is still less 
            // than max distance and the boundary of the dimention has not reached
            if (D != nXTr.getCol()-1 )
            {
                // get the feature vector from both testing and training sample 
                // untill D dimention only
                s1 = subMatrix(nXTr, i,0,i,D);

                s2 = subMatrix(sample,0,0,0,D);
                // get the calculated partial distance
                dis = calculatePartDis(s1,s2);
                D++;

            }
            else
            {
                break;
            }
        }
        // this means calculated distance all the way to whole dimention 
        // and the calculated distance is less than the max distance kept in 
        // K sorted distance so far. 
        if ( D == nXTr.getCol()-1 && dis < maxDis  )
        {
            Kdistance(K-1,0) = dis;
            Kdistance(K-1,1) = nXTr(i,nXTr.getCol()-1);
            Kdistance = ComputeSortedDis(Kdistance);

            // get the current max from the kept K nearest distance so far list
            maxDis = Kdistance(K-1,0);
            D = 0;
        }
    }

    // after going through the finding of K nearest neighbor
    // get the number of sample fall withing that K nearest neighborhood 
    // for each type and assign the class corresponding to the class with 
    // max number of sample found 

    int maxNum = 0;
    int Class = 0;
    Matrix Type0 = getType(Kdistance,0);
    Matrix Type1 = getType(Kdistance,1);
    if (Type0.getRow() > Type1.getRow())
    {
        Class= 0;
    }
    else
    {
        Class = 1;
    }
    return(Class);
}

// this function calculates the partial Minkowski distance to be used for 
// kNN implementation using Partial Minkowski distance
double calculatePartDisMinkowski(Matrix &s1, Matrix &s2, double minK)
{
    double d = 0;
    int col = s1.getCol();

    for ( int i =0; i < col; i++)
    {
        d = d + pow(fabs((s1(0,i) - s2(0,i))), minK);
    }
    d = pow(d, 1/minK);

    return d;
}

// this method build the validating data set for 10-fold cross-validation
// with kNN as classifier
Matrix getTestingData(Matrix &S, Matrix &glassData)
{
    int row = 0;
    int row1 = -1;

    // ge the count of row number without zero value of the sample from the current fold pass that needs to be
    // included in this validating set among the entire "fglass" data set
    for ( int i =0; i < S.getCol(); i++)
    {
        int n = S(0,i);

        // get how many row would be in the validating set
        // disf=regard the zeros as it was to make matrix holding
        // fold to be of same dimention
        if ( n != 0)
        {
            row++;
        }
    }

    Matrix ret(row, glassData.getCol());

    // ge the row number without zero value of the sample from the current fold pass that needs to be
    // included in this validating set among the entire "fglass" data set
    for ( int i = 0; i < S.getCol(); i++)
    {
        int n = S(0,i);

        if ( n != 0)
        {
            row1++;

            Matrix Samp = subMatrix(glassData,n,0,n,glassData.getCol()-1);

            for ( int j = 0; j <Samp.getCol(); j++  )
            {
                // include the corresponding sample from the 
                // normalized "flassdata" set
                ret(row1,j) = Samp(0,j);
            }
        }

    }

    return ret;

}

// this method build the training data set for 10-fold cross-validation
// with kNN as classifier
Matrix getTrainingData(Matrix &foldData, Matrix &glassData, int i )
{
    Matrix samp;
    int R1 = 0;

    // i indicates the fold number that is being consider for serving as validating
    // set in current round of toal of 10 rounds. 

    // in this case i is the first fold so include all other 9 fold excluding
    // first fold to build the training set at current round of toal of 10 rounds.
    if ( i == 0)
    {
        samp = subMatrix(foldData,i+1,0,foldData.getRow()-1,foldData.getCol()-1);

    }

    // in this case i is the last fold so include all other 9 fold excluding
    // last fold to build the training set at current round of toal of 10 rounds.
    else if ( i == foldData.getRow()-1)
    {
        samp = subMatrix(foldData,0,0,foldData.getRow()-2,foldData.getCol()-1);
    }
    // in this case i is anywhre between first and last fold number.
    else
    {
        // so first half of the fold from all fold above current fold
        Matrix first = subMatrix(foldData,0,0,i-1,foldData.getCol()-1);

        // and second half  from all fold below current fold, 
        Matrix second = subMatrix(foldData,i+1,0,foldData.getRow()-1,foldData.getCol()-1);

        //then combine these partial fold to get one whole fold 
        Matrix whole(first.getRow()+second.getRow(), first.getCol());

        int R = 0;
        //use this whole fold to build the training set at current round of toal of 10 rounds.
        for ( R = 0; R < first.getRow(); R++)
        {
            for ( int j =0; j< first.getCol(); j++)
            {
                whole(R,j) = first(R,j);
            }
        }
        while ( R1 < second.getRow() )
        {
            for ( int j =0; j< second.getCol(); j++)
            {
                whole(R,j) = second(R1,j);
            }
            R++;
            R1++;
        }
        samp = whole;
    }

    int row = 0;
    int row1 = -1;

    // ge the count of row number without zero value of the sample from the current fold pass that needs to be
    // included in this validating set among the entire "fglass" data set
    for ( int i = 0; i < samp.getRow(); i++)
    {
        for (int j = 0; j < samp.getCol(); j++)
        {
            int n = samp(i,j);

            if ( n != 0)
            {
                row++;
            }
        }
    }

    Matrix ret(row, glassData.getCol());

    // ge the row number without zero value of the sample from the current fold pass that needs to be
    // included in this validating set among the entire "fglass" data set
    for ( int i = 0; i < samp.getRow(); i++)
    {
        for (int j =0; j < samp.getCol(); j++)
        {
            int n = samp(i,j);
            if ( n != 0)
            {
                row1++;

                Matrix Samp1 = subMatrix(glassData,n,0,n,glassData.getCol()-1);


                for ( int k = 0; k < Samp1.getCol(); k++  )
                {
                    ret(row1,k) = Samp1(0,k);
                }
            }
        }
    }
    return ret;
}

// this function implements kNN using original Euclidean distance to be used to 
// clasiify the testing set using 10 folds cross-validation technique
int KNNClassifierEuclidianFold(Matrix &nXTr, Matrix &sample, int K,int classes)
{
    Matrix distance(nXTr.getRow(),2);
    Matrix samp;
    double sumDistance = 0.0;

    samp = subMatrix(sample, 0,0,0,sample.getCol()-1);

    // calculate the original Euclidean distance between the current tesing sample and each of the 
    // sample in the training set
    for ( int i =0; i < nXTr.getRow(); i++)
    {
        sumDistance = 0.0;
        Matrix samp1 = subMatrix(nXTr, i, 0,i,nXTr.getCol()-2);

        // get the square of the difference distance for each feature of the
        for ( int j =0; j < samp1.getCol(); j++)
        {
            sumDistance = sumDistance + pow((samp(0,j) - samp1(0,j)), 2);
        }         
        sumDistance = sqrt(sumDistance);
        // keep the disance calculated from each training sample along 
        // with their class label info
        distance(i,0) = sumDistance;
        distance(i,1) = nXTr(i, nXTr.getCol()-1);
    }

    // get just the distance part as row to use insersort function to sort the distances
    Matrix DisUnSort = transpose(subMatrix(distance, 0, 0, distance.getRow()-1, 0));

    Matrix DisSort(DisUnSort.getRow(), DisUnSort.getCol()); // this will keep sorted distance
    Matrix DisPos(DisUnSort.getRow(), DisUnSort.getCol());  // this will keep the index 
    // of the sorted distance 

    insertsort(DisUnSort, DisSort, DisPos);
    Matrix Kdistance(K, 2);   // to keep only K nearest distances

    // get only the k nearest distance from the list of all calcualted sorted 
    // distance along with the label for the sample corresponding at that distance apart
    for ( int i =0; i< K; i++)
    {
        int pos = DisPos(0,i);
        int label = distance(pos,1);
        Kdistance(i,0) = DisSort (0,i);
        Kdistance(i,1) = label;
    }

    int maxNum = -1;
    int Class = -1;
    Matrix type;
    int num =0;

    // this is the 7 class category data set which has class label from 1 to 7
    // so for each class type get the number of sample that is of that class and 
    // included in the that K nearest neighborhood distance list, also keep track of the
    // max number seen so far along withe label of the class for which that max number has
    // been seen. Then assign the class corresponding to the class with 
    // max number of sample found at last 
    for ( int i = 1; i <= classes; i++)
    {
        type = getType(Kdistance,i);
        num = type.getRow();
        if (num > maxNum)
        {
            maxNum = num;
            Class = i;
        }
    }
    return(Class);
}
