/* Ryan Wendling
402 final assignment
Zhang Fall 2016

The purpose of this assignment is to develope a multi-layer Artificial Neural Network using backpropagation
with the help of nonlinear sigmoid functions and stochastic gradient descent rule. With it, we will "train"
the program to be able to identify if a mushroom is poisonous or edible with some consistency,
based off its characteristics. This "version-2" is more accurate than the last*/


import java.util.*;
import java.lang.*;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedReader; 
import java.io.FileNotFoundException;


public class ANN {

   // Learning speed for entire program run
   static double learningSpeed = .1;
   // Momentum constant to help update weights
   static double momentumConstant = .5;


   /* Main will use the ANN w/back propagation functions to determine accurate weights
   with which to identify if a mushroom is poisonous or edible. 6000 or so mushrooms will
   be used for testing over 1000s of epochs in order to train the weights. From there, the remaining
   mushrooms will be used to see just how accurate our ANN can identify them. */
   public static void main (String[] args) {
   
      // Read in Mushroom.data file from command line arguements
      File inFile = null;
      if (0 < args.length) {
         inFile = new File(args[0]);
      } else {
          System.err.println("Invalid arguments count, need name of file for data:" + args.length);
          System.exit(0);
      }
      
      // Timer start
      long start = System.nanoTime(); 
      
      // Create structures to hold mushroom data
      double[][] wholeData = fillMushArray(inFile);
      double[][] trainData = new double[6906][23];
      double[][] testData = new double[1218][23];
      
      // Loop through wholeData randomly, dividing values between test and train arrays
      List<Integer> rangeList = new ArrayList<Integer>();
      
      for (int i = 0; i < 8124; i++) {
          rangeList.add(i);
      }
      Collections.shuffle(rangeList);
      int counter = 0;
      int counter2 = 0;
      
      for (int i = 0; i < rangeList.size(); i++) {
         int currentRandomNum = rangeList.get(i);
         
         if (counter < 6906) {
            for (int j = 0; j < 23; j++) { 
	   	      trainData[counter][j] = wholeData[currentRandomNum][j];
            }   
            counter++;
            
         } else {
            for (int j = 0; j < 23; j++) { 
	   	      testData[counter2][j] = wholeData[currentRandomNum][j];
            }   
            counter2++;
         }     
	   }       
      
      // Create random initial weights within double array/matrix
      double[][] weights1 = new double[22][16];
      double[][] weights2 = new double[16][10];
      double[][] weights25 = new double[10][6];
      double[][] weights3 = new double[6][1];

      
      for(int i = 0; i < weights1.length; i++) {
         for(int j = 0; j < weights1[i].length; j++) {
            double aWeightVal = matrixRandWeight();
            weights1[i][j] = aWeightVal;
         }
      }
      for(int i = 0; i < weights2.length; i++) {
         for(int j = 0; j < weights2[i].length; j++) {
            double aWeightVal = matrixRandWeight();
            weights2[i][j] = aWeightVal;
         }
      }
      for(int i = 0; i < weights25.length; i++) {
         for(int j = 0; j < weights25[i].length; j++) {
            double aWeightVal = matrixRandWeight();
            weights25[i][j] = aWeightVal;
         }
      }
      for(int i = 0; i < weights3.length; i++) {
         for(int j = 0; j < weights3[i].length; j++) {
            double aWeightVal = matrixRandWeight();
            weights3[i][j] = aWeightVal;
         }
      } 
      
      System.out.println("3 hidden layers and 1 output layer.");
      System.out.println("Nodes in first layer are 16, nodes in second layer are 10, nodes in third layer are 6, 1 node in output layer.");
      System.out.println("initial learning speed is .1");
      System.out.println("Weight values for W1: " + Arrays.deepToString(weights1));
      System.out.println("Weight values for W2: " + Arrays.deepToString(weights2));
      System.out.println("Weight values for W3: " + Arrays.deepToString(weights25));
      System.out.println("Weight values for W4: " + Arrays.deepToString(weights3));
      
      
      //FORWARD PROPAGATION STARTS HERE:   
      double lastAccuracy = 0;
      double lastLastAccuracy = 0;
      // arrays used for momentum
      double[][] prevWeightChange1 = new double[22][16];
      double[][] prevWeightChange2 = new double[16][10];
      double[][] prevWeightChange25 = new double[10][6];
      double[][] prevWeightChange3 = new double[6][1];
      
      for (int p = 0; p < 5000; p ++) {
         double accuracy = 0;
         
         for(int q = 0; q <  trainData.length; q++) { 
         
            // Grabs the first mushroom and puts it into usable format
            double[][] aMushroom = new double[1][22];
            for(int j = 1; j < trainData[q].length; j++) {
               aMushroom[0][j-1] = trainData[q][j];
            }
      
            // matrix mult and sigmoidify H1
            double[][] result1 = matrixMult(aMushroom, weights1);
            double[][] newResult1 =sigmoid(result1);
      
            // repeat matrix mult and sigmoidify H2
            double[][] result2 = matrixMult(newResult1, weights2);
            double[][] newResult2 =sigmoid(result2);
            
            // repeat matrix mult and sigmoidify H25
            double[][] result25 = matrixMult(newResult2, weights25);
            double[][] newResult25 = sigmoid(result25);
                        
            // repeat matrix mult and sigmoidify H3
            double[][] result3 = matrixMult(newResult25, weights3);
            double[][] newResult3 = sigmoid(result3);
            
            // Get real value in binary form        
            double desiredVal = trainData[q][0];

            // IF RIGHT, STOP ITERATION, OTHERWISE ADJUST EVERYTHING
            if (((newResult3[0][0] > .5) && (desiredVal > .5)) || ((newResult3[0][0] <= .5) && (desiredVal <= .5))) {
               accuracy++;
            } else {    
            
                   
            //BACK PROPOGATION STARTS HERE:
            // stochastic gradient descent rule for output, 
            double outcomeGradVal = stochGradDescOut(newResult3[0][0], desiredVal);
            double[]outcomeGradValArr = new double[1];
            outcomeGradValArr[0] = outcomeGradVal;
         
               // update weights3 as we do the stochastic gradient descent
               for (int j = 0; j < weights3[0].length; j++) {
                  for (int i = 0; i < weights3.length; i++) {
                      weights3[i][j] =  updateWeight(weights3[i][j], outcomeGradVal, newResult3[0][j], i, j, prevWeightChange3);
                  }
               }   
               
               // stochastic gradient descent rule for hidden layer 25 
               double[]hiddenGradVals25 = new double[result25[0].length];
               for (int i = 0; i < result25[0].length; i++) {
                  double aHiddenGradVals25 = stochGradDescHid(newResult25[0][i], outcomeGradValArr, weights3[i]);
                  hiddenGradVals25[i] = aHiddenGradVals25;
               }   
            
               // update weights25 as we do the stochastic gradient descent
               for (int j = 0; j < weights25[0].length; j++) {
                  for (int i = 0; i < weights25.length; i++) {
                     weights25[i][j] =  updateWeight(weights25[i][j], hiddenGradVals25[j], newResult25[0][j], i, j, prevWeightChange25);
                  }
               }               
                         
               // stochastic gradient descent rule for hidden layer 2 
               double[]hiddenGradVals2 = new double[result2[0].length];
               for (int i = 0; i < result2[0].length; i++) {
                  double aHiddenGradVals2 = stochGradDescHid(newResult2[0][i], hiddenGradVals25, weights25[i]);
                  hiddenGradVals2[i] = aHiddenGradVals2;
               }   
            
               // update weights2 as we do the stochastic gradient descent
               for (int j = 0; j < weights2[0].length; j++) {
                  for (int i = 0; i < weights2.length; i++) {
                     weights2[i][j] =  updateWeight(weights2[i][j], hiddenGradVals2[j], newResult2[0][j], i, j, prevWeightChange2);
                  }
               }      
                      
               // stochastic gradient descent rule for hidden layer 1 
               double[]hiddenGradVals1 = new double[result1[0].length];
               for (int i = 0; i < result1[0].length; i++) {
                  double aHiddenGradVals1 = stochGradDescHid(newResult1[0][i], hiddenGradVals2, weights2[i]);
                  hiddenGradVals1[i] = aHiddenGradVals1;
               }   
               
               // update weights1 as we do the stochastic gradient descent   
               for (int j = 0; j < weights1[0].length; j++) {
                 for (int i = 0; i < weights1.length; i++) {
                     weights1[i][j] = updateWeight(weights1[i][j], hiddenGradVals1[j], newResult1[0][j], i, j, prevWeightChange1);
                  }
               }    
            }
         }
         if (p == 50) {
            System.out.println("W2 weights after interval 50: " + Arrays.deepToString(weights2));  
         }
         double percentage = (accuracy/6909)*100;     
         System.out.print("Accuracy is: " + accuracy + " or ");
         System.out.print( String.format( "%.2f", percentage ) + "%  "); 
         System.out.println("for epoch: " + p);
         if ((lastAccuracy == accuracy) && (lastLastAccuracy == lastAccuracy) && (p > 250)) {
            break;
         } else {
         lastLastAccuracy = lastAccuracy;
         lastAccuracy = accuracy; 
         }
         if (p > 1500) {
            break;
         }   
      }
      System.out.println("Training done, look at previous line for last epoch and most recent accuracy");
      System.out.println("time program took, in nanoseconds: " + (System.nanoTime() - start));
      System.out.println("Final training weight values for W1: " + Arrays.deepToString(weights1));
      System.out.println("Final training weight values for W2: " + Arrays.deepToString(weights2));
      System.out.println("Final training weight values for W3: " + Arrays.deepToString(weights25));
      System.out.println("Final training weight values for W4: " + Arrays.deepToString(weights3));
   

      // TESTING ROUND BEGINS HERE:
      double accuracy2 = 0;
      
      for(int q = 0; q <  testData.length; q++) {   
      
         // Grabs the first mushroom and puts it into usable format
         double[][] aMushroom = new double[1][22];
         for(int j = 1; j < testData[q].length; j++) {
            aMushroom[0][j-1] = testData[q][j];
         }
   
         // matrix mult and sigmoidify H1
         double[][] result1 = matrixMult(aMushroom, weights1);
         double[][] newResult1 =sigmoid(result1);
   
         // repeat matrix mult and sigmoidify H2
         double[][] result2 = matrixMult(newResult1, weights2);
         double[][] newResult2 =sigmoid(result2);
         
         // repeat matrix mult and sigmoidify H25
         double[][] result25 = matrixMult(newResult2, weights25);
         double[][] newResult25 =sigmoid(result25);
         
         // repeat matrix mult and sigmoidify H3
         double[][] result3 = matrixMult(newResult25, weights3);
         double[][] newResult3 = sigmoid(result3);
         
         // Get real value in binary form        
         double desiredVal = testData[q][0];

         // IF RIGHT, UP ACCURACY
         if (((newResult3[0][0] > .5) && (desiredVal > .5)) || ((newResult3[0][0] <= .5) && (desiredVal <= .5))) {
            accuracy2++;
         }
      }    
      double percentage2 = (accuracy2/1218)*100;   
      System.out.print("Testing accuracy is: " + accuracy2 + " or ");
      System.out.println( String.format( "%.2f", percentage2 ) + "%");
   }
       
      
   /* returns a random weight between -1 and 1. To be used for
   the initial weight value matrices. */
   public static double matrixRandWeight() {
   
      Random percent = new Random();
      int number = percent.nextInt(10);
      double result = number / 10.0;
      Random sign = new Random();
      int num = sign.nextInt(2)+1;
      if (num == 1) {
         result *= -1.0;
      }   
      return result;
   }
  
   
   /* performs the stochastic gradient descent for the one output value we have,
   returns the gradient value as a double. */
   public static double stochGradDescOut(double oldOutcome, double desiredVal) {
      
      double part1 = (1 - oldOutcome);
      double part2 = (desiredVal - oldOutcome);
      double next = oldOutcome * part1;
      double newGradient = next * part2;
            return newGradient;
   }
   
   
   /* performs the stochastic gradient descent for the hidden values we have,
   returns the gradient value as a double. */
   public static double stochGradDescHid(double oldHiddenVal, double[] prevGradValues, double[]weights) {
   
       double weightGradSums = 0;
       for (int i = 0; i < weights.length; i++) {
           weightGradSums += prevGradValues[i] * weights[i];
       }    
       double newGradient = oldHiddenVal * (1 - oldHiddenVal) * weightGradSums;
       return newGradient;  
   }
     
   
   /* Functioni that performs the sigmoid function to the incoming data, after it has been matrix
   multiplied with the weights. This is performed at every hidden layer during forward propagation. */
   public static double[][] sigmoid(double[][] outcome) {
   
      double [][] result = new double[outcome.length][outcome[0].length];
      
      for(int i = 0; i < outcome.length; i++) {
      
         for(int j = 0; j < outcome[i].length; j++) {
            double initSum = (outcome[i][j]);
            double negInitSum = initSum;
            negInitSum *= -1.0;
            double bottom = 1 + (Math.exp(negInitSum));
            result[i][j] = 1 / bottom;
         }
      }
      return result;
   }
   
   
   /* Function that updates one of the weights at a certain section (ex: W1, W2, W3, etc.).
   The input values are the old weight from that section, along with the column's
   pre-gradient value and current column's gradient value.*/
   public static double updateWeight(double aWeight, double prevGrad, double thisGrad, int spotx, int spoty, double [][]prevWeightChange) {
   
      double newWeight = aWeight + learningSpeed*prevGrad*thisGrad + momentumConstant*prevWeightChange[spotx][spoty];
      prevWeightChange[spotx][spoty] = learningSpeed*prevGrad*thisGrad;
      return newWeight;
   }
   
   
   /* Matrix multiplication function. Takes in two seperate multi-dimensional arrays of doubles.
   Returns the resulting matrix to the caller. */
   public static double[][] matrixMult(double[][] inputs, double[][]weights) {
   
      double [][] result = new double[inputs.length][weights[0].length];
      
      /* Loop through each and get product, then sum up and store the value */
      for (int i = 0; i < inputs.length; i++) { 
          for (int j = 0; j < weights[0].length; j++) { 
              for (int k = 0; k < inputs[0].length; k++) { 
                  result[i][j] += inputs[i][k] * weights[k][j];
              }
          }
      }
      return result;
   }
   
   
   /* Function that takes in the "mushroom.data" file and transfers the contents into a double array,
   where each line is a mushroom. Will return the filled array. */
   public static double[][] fillMushArray(File inFile) {
   
      double[][] wholeData = new double[8124][23];
      
      // Read through mushroom data, converting chars to values along spectrum
      // depending on available options and assigning into wholeDataList. Normalizes input data
      BufferedReader reader = null;
      
      try {
          reader = new BufferedReader(new FileReader(inFile));
          String textLine;
          int mushroomNumber = 0;
             
          while ((textLine = reader.readLine()) != null) {
              int characteristicSpot = 0;
              
              for (int i = 0; i < textLine.length(); i++){
                  char c = textLine.charAt(i);  
                  // disregards ','   
                  if (c != 44) {    
                     double normalizedVal = 0;
                     if (i * 2 == 0 ) {
                        if (c == 112) {
                           normalizedVal = 0;
                        } else {
                           normalizedVal = 1;
                        }      
                     } else if (i == 2) {
                        switch(c) {
                           case 'b' :
                           normalizedVal = 1;
                           break;
                           case 'c' :
                           normalizedVal = 2;
                           break;
                           case 'x' :
                           normalizedVal = 3;
                           break;
                           case 'f' :
                           normalizedVal = 4;
                           break;
                           case 'k' :
                           normalizedVal = 5;
                           break;
                           case 's' :
                           normalizedVal = 6;
                           break;
                        }
                     } else if (i == 4) {   
                        switch(c) {
                           case 'f' :
                           normalizedVal = 1;
                           break;
                           case 'g' :
                           normalizedVal = 2;
                           break;
                           case 'y' :
                           normalizedVal = 3;
                           break;
                           case 's' :
                           normalizedVal = 4;
                           break;
                        }
                     } else if (i == 6) {   
                        switch(c) {
                           case 'n' :
                           normalizedVal = 1;
                           break;
                           case 'b' :
                           normalizedVal = 2;
                           break;
                           case 'c' :
                           normalizedVal = 3;
                           break;
                           case 'g' :
                           normalizedVal = 4;
                           break;
                           case 'r' :
                           normalizedVal = 5;
                           break;
                           case 'p' :
                           normalizedVal = 6;
                           break;
                           case 'u' :
                           normalizedVal = 7;
                           break;
                           case 'e' :
                           normalizedVal = 8;
                           break;
                           case 'w' :
                           normalizedVal = 9;
                           break;
                           case 'y' :
                           normalizedVal = 10;                         
                           break;
                        }   
                     } else if (i == 8) {   
                        switch(c) {
                           case 't' :
                           normalizedVal = 1;
                           break;
                           case 'f' :
                           normalizedVal = 2;
                           break;
                        }
                     } else if (i == 10) {   
                        switch(c) {
                           case 'a' :
                           normalizedVal = 1;
                           break;
                           case 'l' :
                           normalizedVal = 2;
                           break;
                           case 'c' :
                           normalizedVal = 3;
                           break;
                           case 'y' :
                           normalizedVal = 4;
                           break;                        
                           case 'f' :
                           normalizedVal = 5;
                           break;
                           case 'm' :
                           normalizedVal = 6;
                           break;
                           case 'n' :
                           normalizedVal = 7;
                           break;
                           case 'p' :
                           normalizedVal = 8;
                           break;
                           case 's' :
                           normalizedVal = 9;
                           break;
                        }                        
                     } else if (i == 12) {   
                        switch(c) {
                           case 'a' :
                           normalizedVal = 1;
                           break;
                           case 'd' :
                           normalizedVal = 2;
                           break;
                           case 'f' :
                           normalizedVal = 3;
                           break;
                           case 'n' :
                           normalizedVal = 4;
                           break;
                        }
                     } else if (i == 14) {   
                        switch(c) {
                           case 'c' :
                           normalizedVal = 1;
                           break;
                           case 'w' :
                           normalizedVal = 2;
                           break;
                           case 'd' :
                           normalizedVal = 3;
                           break;
                        }
                     } else if (i == 16) {   
                        switch(c) {
                           case 'b' :
                           normalizedVal = 1;
                           break;
                           case 'n' :
                           normalizedVal = 2;
                           break;
                        }
                     } else if (i == 18) {   
                        switch(c) {
                           case 'k' :
                           normalizedVal = 1;
                           break;
                           case 'n' :
                           normalizedVal = 2;
                           break;
                           case 'b' :
                           normalizedVal = 3;
                           break;
                           case 'h' :
                           normalizedVal = 4; 
                           break;                       
                           case 'g' :
                           normalizedVal = 5;
                           break;
                           case 'r' :
                           normalizedVal = 6;
                           break;
                           case 'o' :
                           normalizedVal = 7;
                           break;
                           case 'p' :
                           normalizedVal = 8;
                           break;
                           case 'u' :
                           normalizedVal = 9;
                           break;
                           case 'e' :
                           normalizedVal = 10;
                           break;
                           case 'w' :
                           normalizedVal = 11;
                           break;
                           case 'y' :
                           normalizedVal = 12;                          
                           break;
                        }                                                                        
                     } else if (i == 20) {   
                        switch(c) {
                           case 'e' :
                           normalizedVal = 1;
                           break;
                           case 't' :
                           normalizedVal = 2;
                           break;
                        }
                     } else if (i == 22) {   
                        switch(c) {
                           case 'b' :
                           normalizedVal = 1;
                           break;
                           case 'c' :
                           normalizedVal = 2;
                           break;
                           case 'u' :
                           normalizedVal = 3;
                           break;
                           case 'e' :
                           normalizedVal = 4;  
                           break;                      
                           case 'z' :
                           normalizedVal = 5;
                           break;
                           case 'r' :
                           normalizedVal = 6;
                           break;
                           case '?' :
                           normalizedVal = 7;                         
                           break;
                        }
                     } else if (i == 24) {   
                        switch(c) {
                           case 'f' :
                           normalizedVal = 1;
                           break;
                           case 'y' :
                           normalizedVal = 2;
                           break;
                           case 'k' :
                           normalizedVal = 3;
                           break;
                           case 's' :
                           normalizedVal = 4;                                               
                           break;
                        }                        
                     } else if (i == 26) {   
                        switch(c) {
                           case 'f' :
                           normalizedVal = 1;
                           break;
                           case 'y' :
                           normalizedVal = 2;
                           break;
                           case 'k' :
                           normalizedVal = 3;
                           break;
                           case 's' :
                           normalizedVal = 4;                                               
                           break;
                        }   
                     } else if (i == 28) {   
                        switch(c) {
                           case 'n' :
                           normalizedVal = 1;
                           break;
                           case 'b' :
                           normalizedVal = 2;
                           case 'c' :
                           normalizedVal = 3;
                           break;
                           case 'g' :
                           normalizedVal = 4;
                           break;                        
                           case 'o' :
                           normalizedVal = 5;
                           break;
                           case 'p' :
                           normalizedVal = 6;
                           break;
                           case 'e' :
                           normalizedVal = 7;
                           break;
                           case 'w' :
                           normalizedVal = 8;
                           break;
                           case 'y' :
                           normalizedVal = 9;                        
                           break;
                        }
                     } else if (i == 30) {   
                        switch(c) {
                           case 'n' :
                           normalizedVal = 1;
                           break;
                           case 'b' :
                           normalizedVal = 2;
                           break;
                           case 'c' :
                           normalizedVal = 3;
                           break;
                           case 'g' :
                           normalizedVal = 4; 
                           break;                       
                           case 'o' :
                           normalizedVal = 5;
                           break;
                           case 'p' :
                           normalizedVal = 6;
                           break;
                           case 'e' :
                           normalizedVal = 7;
                           break;
                           case 'w' :
                           normalizedVal = 8;
                           break;
                           case 'y' :
                           normalizedVal = 9;                        
                           break;
                        }                        
                     } else if (i == 32) {   
                        switch(c) {
                           case 'p' :
                           normalizedVal = 1;
                           break;
                           case 'u' :
                           normalizedVal = 2;                                            
                           break;
                        } 
                      } else if (i == 34) {   
                        switch(c) {
                           case 'n' :
                           normalizedVal = 1;
                           break;
                           case 'o' :
                           normalizedVal = 2; 
                           break;  
                           case 'w' :
                           normalizedVal = 3;
                           break;
                           case 'y' :
                           normalizedVal = 4;                                          
                           break;
                        }                        
                      } else if (i == 36) {   
                        switch(c) {
                           case 'n' :
                           normalizedVal = 1;
                           break;
                           case 'o' :
                           normalizedVal = 2; 
                           break;  
                           case 't' :
                           normalizedVal = 3;                                       
                           break;
                        }                        
                     } else if (i == 38) {     
                        switch(c) {
                           case 'c' :
                           normalizedVal = 1;
                           break;
                           case 'e' :
                           normalizedVal = 2;
                           break;
                           case 'f' :
                           normalizedVal = 3;
                           break;
                           case 'l' :
                           normalizedVal = 4;  
                           break;                      
                           case 'n' :
                           normalizedVal = 5;
                           break;
                           case 'p' :
                           normalizedVal = 6;
                           break;
                           case 's' :
                           normalizedVal = 7;
                           break;
                           case 'z' :
                           normalizedVal = 8;                     
                           break;
                        }  
                     } else if (i == 40) {     
                        switch(c) {
                           case 'k' :
                           normalizedVal = 1;
                           break;
                           case 'n' :
                           normalizedVal = 2;
                           break;
                           case 'b' :
                           normalizedVal = 3;
                           break;
                           case 'h' :
                           normalizedVal = 4; 
                           break;                       
                           case 'r' :
                           normalizedVal = 5;
                           break;
                           case 'o' :
                           normalizedVal = 6;
                           break;
                           case 'u' :
                           normalizedVal = 7;
                           break;
                           case 'w' :
                           normalizedVal = 8; 
                           break;  
                           case 'y' :
                           normalizedVal = 9;                                              
                           break;
                        }
                     } else if (i == 42) {     
                        switch(c) {
                           case 'a' :
                           normalizedVal = 1;
                           break;
                           case 'c' :
                           normalizedVal = 2;
                           break;
                           case 'n' :
                           normalizedVal = 3;
                           break;
                           case 's' :
                           normalizedVal = 4;   
                           break;                     
                           case 'v' :
                           normalizedVal = 5;
                           break;
                           case 'y' :
                           normalizedVal = 6;                                             
                           break;
                        }                                                
                     } else if (i == 44) {     
                        switch(c) {
                           case 'g' :
                           normalizedVal = 1;
                           break;
                           case 'l' :
                           normalizedVal = 2;
                           break;
                           case 'm' :
                           normalizedVal = 3;
                           break;
                           case 'p' :
                           normalizedVal = 4; 
                           break;                       
                           case 'u' :
                           normalizedVal = 5;
                           break;
                           case 'w' :
                           normalizedVal = 6;  
                           break;  
                           case 'd' :
                           normalizedVal = 7;                                          
                           break;
                        }                                                                         
                     } else {
                     }                                                                     
                     wholeData[mushroomNumber][characteristicSpot] = normalizedVal;
                     characteristicSpot++;
                  }  
              }
              mushroomNumber++;
          }
      } catch (FileNotFoundException e) {
          e.printStackTrace();
      } catch (IOException e) {
          e.printStackTrace();
      } finally {
          try {
              if (reader != null) {
                  reader.close();
              }
          } catch (IOException e) {
          }
      }
      return wholeData;
   }
}   