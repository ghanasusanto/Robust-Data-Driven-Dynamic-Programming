#define ILOUSESTL
using namespace std;

#include <ilcplex/ilocplex.h>
#include <vector>
#include "mex.h"

ILOSTLBEGIN
typedef IloArray<IloNumArray>    NumMatrix;
typedef IloArray<IloNumVarArray> NumVarMatrix;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    if (nrhs != 6) {
        mexErrMsgTxt("Six input arguments required.");
    } else if (nlhs > 4) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    size_t N = mxGetScalar(prhs[0]);
    size_t d = mxGetScalar(prhs[1]);
    double* out_value;
    double* out_u;
    double* out_status;
    
    IloEnv env;
    try {
        IloModel mod(env);
        IloNumArray q(env,N);
        IloNum gamma = mxGetScalar(prhs[4]);
        IloNum rho = mxGetScalar(prhs[5]);
        
        
        NumMatrix next_xi(env,N);
        IloNumVar mu(env, -IloInfinity, IloInfinity);
        IloNumVar lambda(env, 0, IloInfinity);
        IloNumVarArray y(env, N, -IloInfinity, IloInfinity);
        IloNumVarArray z(env, N, -IloInfinity, IloInfinity);
        IloNumVarArray rhs(env, N, 0, IloInfinity);
        IloNumVarArray u(env,d,-IloInfinity,IloInfinity);
        IloExpr value(env);
        mod.add(IloSum(u) == 1);
        value += lambda*gamma - mu;
        for (int i=0; i < N; i++) {
            next_xi[i] = IloNumArray(env,d);
            for (int j=0; j < d; j++) {
                next_xi[i][j] = mxGetPr(prhs[3])[j*N+i];
            }
            q[i] = mxGetPr(prhs[2])[i];
            value += -2*q[i]*y[i] + 2*lambda*q[i];
            
            mod.add(mu + z[i] <=  lambda);
            mod.add(rho*IloSquare(IloScalProd(u,next_xi[i])) - IloScalProd(u,next_xi[i]) <= z[i]);
            mod.add ( IloSquare(y[i]) + 0.25*IloSquare(z[i] + mu) <= IloSquare(rhs[i]) );
            mod.add ( rhs[i] <= 0.5 * (2*lambda - z[i] - mu) );
            
        }
        mod.add(IloMinimize(env, value));
        value.end();
        IloCplex cplex(mod);
        cplex.setOut(env.getNullStream()); 
        cplex.solve();
        plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
        plhs[1] = mxCreateDoubleMatrix(d,1,mxREAL);
        plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
        out_value = mxGetPr(plhs[0]);
        out_u = mxGetPr(plhs[1]);
        out_status = mxGetPr(plhs[2]);
        *out_value = cplex.getObjValue();
        *out_status = cplex.getStatus();
        for (int i=0;i < d; i++) {
            out_u[i] = cplex.getValue(u[i]);
        }
    }
    catch (IloException& ex) {
        mexErrMsgTxt ("Error robust");
    }
    catch (...) {
        mexErrMsgTxt ("Error");
    }
    env.end();
    
    return;
}