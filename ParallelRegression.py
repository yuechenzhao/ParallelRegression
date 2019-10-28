# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from operator import add
from time import time
from pyspark import SparkContext


def readData(input_file,spark_context):
    """  Read data from an input file and return rdd containing pairs of the form:
                         (x,y)
         where x is a numpy array and y is a real value. The input file should be a 
         'comma separated values' (csv) file: each line of the file should contain x
         followed by y. For example, line:

         1.0,2.1,3.1,4.5

         should be converted to tuple:
        
         (array(1.0,2.1,3.1),4.5)
    """ 
    return spark_context.textFile(input_file)\
        	.map(lambda line: line.split(','))\
        	.map(lambda words: (words[:-1],words[-1]))\
        	.map(lambda inp: (np.array([ float(x) for x in inp[0]]),float(inp[1])))

def readBeta(input):
    """ Read a vector β from CSV file input
    """
    with open(input,'r') as fh:
        str_list = fh.read().strip().split(',')
        
        return np.array( [float(val) for val in str_list] )           

def writeBeta(output,beta):
    """ Write a vector β to a CSV file ouptut
    """
    with open(output,'w') as fh:
        fh.write(','.join(map(str, beta.tolist()))+'\n')
    
def estimateGrad(fun,x,delta):
     """ Given a real-valued function fun, estimate its gradient numerically.
     """
     d = len(x)
     grad = np.zeros(d)
     for i in range(d):
         e = np.zeros(d)
         e[i] = 1.0
         grad[i] = (fun(x+delta*e) - fun(x))/delta
     return grad

def lineSearch(fun,x,grad,a=0.2,b=0.6):
    """ Given function fun, a current argument x, and gradient grad, 
        perform backtracking line search to find the next point to move to.
        (see Boyd and Vandenberghe, page 464).

        Parameters a,b  are the parameters of the line search.

        Given function fun, and current argument x, and gradient  ∇fun(x), the function finds a t such that
        fun(x - t * grad) <= fun(x) - a t <∇fun(x),∇fun(x)>

        The return value is the resulting value of t.
    """
    t = 1.0
    while fun(x-t*grad) > fun(x)- a * t *np.dot(grad,grad):
        t = b * t
    return t 
    
def predict(x,beta):
    """ Given vector x containing features and parameter vector β, 
        return the predicted value: 

                        y = <x,β>   
    """
    
    #count_x=x.size
    #count_b=beta.size
    #x=np.reshape(x,(count_x,1))
    #beta=np.reshape(beta,(count_b,1))
    
    return np.dot(x,beta)

def f(x,y,beta):
    """ Given vector x containing features, true label y, 
        and parameter vector β, return the square error:

                 f(β;x,y) =  (y - <x,β>)^2	
    """
    return (y-predict(x,beta))**2

def localGradient(x,y,beta):
    """ Given vector x containing features, true label y, 
        and parameter vector β, return the gradient ∇f of f:

                ∇f(β;x,y) =  -2 * (y - <x,β>) * x	

        with respect to parameter vector β.
        The return value is  ∇f.
    """
    
    #cnt_x=x.size
    #x=np.reshape(x,(1,cnt_x))
    #return np.dot(-2*(y-predict(x,beta)),x)
    return np.dot(-x, 2*(y-predict(x,beta)))
def F(data,beta,lam = 0):
    """  Compute the regularized mean square error:

             F(β) = 1/n Σ_{(x,y) in data}    f(β;x,y)  + λ ||β ||_2^2   
                  = 1/n Σ_{(x,y) in data} (y- <x,β>)^2 + λ ||β ||_2^2 

         where n is the number of (x,y) pairs in RDD data. 

         Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta: vector β
            - lam:  the regularization parameter λ

         The return value is F(β).
    
    """
    n=data.count()
   

    penalty=lam*np.sum(np.square(beta))
    
    error=data.map(lambda pair:f(pair[0],pair[1],beta)).reduce(lambda a,b:a+b)
    
    reg=(1/n)*error+penalty
    return reg
    
def gradient(data,beta,lam = 0):
    """ Compute the gradient  ∇F of the regularized mean square error 
                F(β) = 1/n Σ_{(x,y) in data} f(β;x,y) + λ ||β ||_2^2   
                     = 1/n Σ_{(x,y) in data} (y- <x,β>)^2 + λ ||β ||_2^2   
 
        where n is the number of (x,y) pairs in data. 

        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β
             - lam:  the regularization parameter λ

        The return value is an array containing ∇F.
    """
    penalty=2*lam*beta
   
    number=data.count()
    lg=data.map(lambda pair:localGradient(pair[0],pair[1],beta)).reduce(lambda x,y: x+y)
    
    return lg*(1.0/number)+penalty
    
    
def test(data,beta):
    """ Compute the mean square error  

        	 MSE(β) =  1/n Σ_{(x,y) in data} (y- <x,β>)^2

        of parameter vector β over the dataset contained in RDD data, where n is the size of RDD data.
        
        Inputs are:             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β

        The return value is MSE(β).  
    """
    return F(data,beta)       

def train(data,beta_0, lam,max_iter,eps):
    """ Perform gradient descent:

        to  minimize F given by
  
             F(β) = 1/n Σ_{(x,y) in data} f(β;x,y) + λ ||β ||_2^2   

        where
             - data: an rdd containing pairs of the form (x,y)
             - beta_0: the starting vector β
             - lam:  is the regularization parameter λ
             - max_iter: maximum number of iterations of gradient descent
             - eps: upper bound on the l2 norm of the gradient
             - a,b: parameters used in backtracking line search

        The function performs gradient descent with a gain found through backtracking
        line search. That is it computes
                   
                   β_k+1 = β_k - γ_k ∇F(β_k) 
        	
        where the gain γ_k is given by
        
        	  γ_k = lineSearch(F,β_κ,∇F(β_k))

        and terminates after max_iter iterations or when ||∇F(β_k)||_2<ε.   

        The function returns:
             -beta: the trained β, 
             -gradNorm: the norm of the gradient at the trained β, and
             -k: the number of iterations performed
    """
    start_time=time()
    error=float('inf')
    
    cnt=0
    beta_k=beta_0
    while error>=eps and cnt<max_iter:
        F_beta=F(data,beta_k,lam)
        grad=gradient(data,beta_k,lam)
        lambda_k=lineSearch(lambda x:F(data,x,lam),beta_k,grad)
        beta_k=beta_k-np.dot(lambda_k,grad)
        error=np.dot(grad,grad)**0.5
        cnt+=1
        time_elaps=time()-start_time
        
        print('@@@Iterations: %d,time elapsed: %fs,present function value: %f, present norm: %f'% (cnt,time_elaps,float(F_beta),float(error)))
    return beta_k,error,cnt
def prepare(data):
    """ Prepare data for aggregating terms,
        
        where
            - data: an rdd containing pairs of the form (x, y)

        Each (x, y) pair is mapped to 
        
            (x*transpose(x), y*x)

        The function returns:
            - prepared_data: an rdd containing pairs of the form (x*transpose(x), y*x)
    
    """
    #def mapper(x):
    #    lst=[]
    #    for i in range(len(x[0])):
    #      lst.append((x[0][i],x[1]))
    #    return lst
    
    #rdd=data.flatMap(mapper).map(lambda pair:(pair[0]*pair[0].T,pair[1]*pair[0])).cache()
    
    rdd=data.map(lambda pair: (np.dot(pair[0],np.transpose(pair[0])),np.dot(pair[1],pair[0]))).cache()
    return rdd

def aggregate(data, lam):
    """
        Aggregate terms to a matrix and a vector, which form a system of linear equations,

        where
            - data: an rdd containing pairs of the form (x_i, y_i)
            - lam: hyperparameter used in calculations, λ

        This function uses input data to generate the matrix 
            
            1/n*transpose(X)*X + λI

        and the vector

            1/n*transpose(X)*y.

        where n is the number of tuples, i.e. data samples and

            X = [x_i] i = 1, ..., n is an n by d matrix and 
            y = [y_i] i = 1, ..., n is a vector
        
        both with real entries.

        The function returns:
            - aggregated_data: the tuple (1/n*transpose(X)*X + λI, 1/n*transpose(X)*y), in given order
    """
   
    
    
    
    mydata=prepare(data)
    foo=mydata.take(1)
    
    cnt=foo[0][0].size

    i=np.eye(cnt)
    #temp=lam*i
    #x=mydata.map(lambda pair:(pair[0])).collect()
    
    
    #t1=1/cnt*x+temp
    #t1=np.array(t1)
    #y=mydata.map(lambda pair:pair[1]).collect()
    #y=np.array(y)
    #t2=y/cnt
    #t2=np.array(t2)
    #res=mydata.map(lambda pair:(pair[0]*1/cnt+temp,pair[1]/cnt))
    res1=mydata.map(lambda x:x[0]).reduce(lambda x,y:x+y)
    t1=res1*1/cnt+lam*i
    res2=mydata.map(lambda x:x[1]).reduce(lambda x,y:x+y)
    t2=res2/cnt
    return (t1,t2)
    
    
def solve_beta(data, lam):
    """ Solve the linear system of equations:

        (1/n*transpose(X)*X + λI)*β = 1/n*transpose(X)*y

        where
            - data: an rdd containing pairs of the form (x_i, y_i)
            - lam: hyperparameter λ used in calculations 


        The function returns:
            - beta: β computed via the numpy "linalg.solve" linear system solver
    """
    start = time()
    print("Aggregating data...")
    A,b = aggregate(data, lam)
    agg_time = time()
    print("...done. Aggregation time:",agg_time-start)
    print("Solving linear system...")
    return np.linalg.solve(A,b)
    sol_time = time()
    print("...done. System solution time:",sol_time-agg_time) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Ridge Regression.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--traindata',default=None, help='Input file containing (x,y) pairs, used to train a linear model.')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a linear model.')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing).')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ.')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations.')
    parser.add_argument('--eps', type=float, default=0.01, help='ε-tolerance. If ||∇F(β_k)||_2<ε, i.e., the Euclidan norm of the gradient is smaller than ε, gradient descent terminates.')
    parser.add_argument('--N',type=int,default=20,help='Number of partitions')
    parser.add_argument('--solver',default='GD',choices=['GD', 'LS'],help='GD learns β via gradient descent, LS learns β by solving a linear system of equations')
        
    verbosity_group=parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose',dest='verbose',action='store_true')
    verbosity_group.add_argument('--silent',dest='verbose',action='store_false')
    parser.set_defaults(verbose=False)
       
    args = parser.parse_args()
    
    sc = SparkContext(appName='Parallel Ridge Regression')
    if not args.verbose:
        sc.setLogLevel("ERROR")
    beta = None
    
    if args.traindata is not None:
        # Train a linear model β from data with regularization parameter λ, and store it in beta
        print('Reading training data from',args.traindata)
        data = readData(args.traindata,sc)
        data = data.repartition(args.N).cache()
        
        x,y = data.take(1)[0]
        beta0 = np.zeros(len(x))
        
        if args.solver == 'GD':
            start = time()
            print('Training on data from',args.traindata,'with lambda =',args.lam,', eps =',args.eps,', max iter = ',args.max_iter)
            beta, gradNorm, k = train(data,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps) 
            print('Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps, 'Training time:', time()-start)
            print('Saving trained beta in',args.beta)
            writeBeta(args.beta,beta)
        
        else:
            start = time()
            print('Solving the linear system for beta on data from',args.traindata,'with lam =',args.lam)
            beta = solve_beta(data, args.lam)
            print(beta)
            print('Training time:',time()-start)
            print('Saving solved beta in',  args.beta)
            writeBeta(args.beta, beta)
     
    if args.testdata is not None:
        # Read beta from args.beta, and evaluate its MSE over data
        print('Reading test data from',args.testdata)
        data = readData(args.testdata,sc)
	
        data = data.repartition(args.N).cache()
        
        print('Reading beta from',args.beta)
        beta = readBeta(args.beta)

        print('Computing MSE on data',args.testdata)
        MSE = test(data,beta)
        print('MSE is:', MSE)

