package com.simple.nn;

import java.util.Arrays;

public class NN {
    private int nNodes;

    public NN() {

    }

    public void forwardPropagation(){

    }
    
    public void run(){

        double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] Y = {{0}, {1}, {1}, {0}};

        int m = 4;
        int nodes = 400;

        X = op.T(X);
        Y = op.T(Y);

        double[][] W1 = op.random(nodes, 2);
        double[][] b1 = new double[nodes][m];

        double[][] W2 = op.random(1, nodes);
        double[][] b2 = new double[1][m];

        for (int i = 0; i < 4000; i++) {
            // Foward Prop
            // LAYER 1
            double[][] Z1 = op.add(op.dot(W1, X), b1);
            double[][] A1 = op.sigmoid(Z1);

            //LAYER 2
            double[][] Z2 = op.add(op.dot(W2, A1), b2);
            double[][] A2 = op.sigmoid(Z2);

            double cost = op.cross_entropy(m, Y, A2);
            //costs.getData().add(new XYChart.Data(i, cost));

            // Back Prop
            //LAYER 2
            double[][] dZ2 = op.subtract(A2, Y);
            double[][] dW2 = op.divide(op.dot(dZ2, op.T(A1)), m);
            double[][] db2 = op.divide(dZ2, m);

            //LAYER 1
            double[][] dZ1 = op.multiply(op.dot(op.T(W2), dZ2), op.subtract(1.0, op.power(A1, 2)));
            double[][] dW1 = op.divide(op.dot(dZ1, op.T(X)), m);
            double[][] db1 = op.divide(dZ1, m);

            // G.D
            W1 = op.subtract(W1, op.multiply(0.01, dW1));
            b1 = op.subtract(b1, op.multiply(0.01, db1));

            W2 = op.subtract(W2, op.multiply(0.01, dW2));
            b2 = op.subtract(b2, op.multiply(0.01, db2));

            if (i % 400 == 0) {
                System.out.println("==============");
                System.out.println("Cost = " + cost);
                System.out.println("Predictions = " + Arrays.deepToString(A2));
            }
        }
    }

    public static void main(String[] args) {
        NN nn = new NN();
        nn.run();
    }

}
