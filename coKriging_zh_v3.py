__author__ = 'zhanghao'

import numpy as np
from numpy.matlib import rand, zeros, ones, empty, eye
import copy
from random import Random,sample
import inspyred
from inspyred import ec
from time import time
from scipy.optimize import minimize
from krige import kriging
from scipy.stats import norm
from samplingplan import samplingplan


class coKriging():
    def __init__(self, Xc, yc, Xe, ye):

        #create the data array
        self.Xc = np.atleast_2d(Xc)

        try:
            yc.shape[1]
        except:
            self.yc = np.atleast_2d(yc).T
        else:
            self.yc = yc

        try:
            ye.shape[1]
        except:
            self.ye = np.atleast_2d(ye).T
        else:
            self.ye = ye

        self.nc = self.Xc.shape[0]
        self.Xe = np.atleast_2d(Xe)
        self.ne = self.Xe.shape[0]
        self.Xd = []
        self.yd = []
        self.yc_xe = []

        self.y = np.vstack((self.yc, self.ye))

        self.yc = self.yc.flatten()

        self.normRange= []
        self.thetamin = 1e-2
        self.thetamax = 1
        self.pmin = 2.
        self.pmax = 2.
        self.rhomin = -2.
        self.rhomax = 2.

        self.LnDetPsid = None

        #self.reorder_data() #not sure here

        self.k = self.Xc.shape[1]

        self.rho = 0.
        self.thetad = np.ones(self.k)
        self.thetac = np.ones(self.k)

        self.pd = np.ones(self.k) * 2.
        self.pc = np.ones(self.k) * 2.

        self.mu = None
        self.muc = None
        self.mud = None
        self.sigmaSqrd = None
        self.sigmaSqrc = None

        self.Psic_Xc = None
        self.UPsic_Xc = None
        self.Psic_Xe = None
        self.UPsic_Xe = None
        self.Psic_XcXe = None
        self.Psic_XeXc = None
        self.Psid_Xe = None
        self.UPsid_Xe = None

        self.NegLnLike = None
        self.LnDetPsid = None

        self.cheapKriging = None
        self.error = []

    # ==========================================================
    # in order to eliminate the affection of unit, here we want to normalize input X
    def normX(self,X):
        X = copy.deepcopy(X)
        for i in range(self.k):
            X[i] = (X[i] - self.normRange[i][0]) / float(self.normRange[i][1] - self.normRange[i][0])
        return X

    def inversenormX(self,X):
        X = copy.deepcopy(X)
        for i in range(self.k):
            X[i] = (X[i] * float(self.normRange[i][1] - self.normRange[i][0])) + self.normRange[i][0]
        return X

    # get the normalized data X
    def normalizeData(self):
        for i in range(self.k):
            self.normRange.append([min(self.Xc[:,i]),max(self.Xc[:,i])])

        for i in range(self.nc):
            self.Xc[i] = self.normX(self.Xc[i])

        for i in range(self.ne):
            self.Xe[i] = self.normX(self.Xe[i])


    # ======================== matrix calculation ===========================

    # now we calculate the |distance| between two points
    def updateData(self):
        self.one = np.ones((len(self.yc) + len(self.ye), 1))
        self.nc = self.Xc.shape[0]
        self.ne = self.Xe.shape[0]
        self.distance_Xc()
        self.distance_Xe()
        self.distance_XcXe()
        self.distance_XeXc()

    def updateModel(self):
        '''
        this function rebuilds the Psi matrix to reflect new data or a change in hyperparameters
        '''
        try:
            self.updatePsi()
        except Exception as err:
            raise Exception("fail to update PSI matrix")

    # now we train the cheap data to get the cheap kriging model

    # now we calculate the distance in dataset Xc, Xe, and XcXe
    def distance_Xc(self):
        self.distanceXc = np.zeros((self.nc,self.nc,self.k))
        for i in range(self.nc):
            for j in range(i+1, self.nc):
                self.distanceXc[i][j] = np.abs((self.Xc[i]-self.Xc[j]))



    def distance_Xe(self):
        self.distanceXe = np.zeros((self.ne,self.ne,self.k))
        for i in range(self.ne):
            # for j in range(i+1,self.ne):
            for j in range(self.ne):
                self.distanceXe[i][j] = np.abs((self.Xe[i] - self.Xe[j]))

    #notice that here we want to set PsicXcXe nc * ne, this is an asymmetric matrix
    def distance_XcXe(self):
        self.distanceXcXe = np.zeros((self.nc,self.ne,self.k))
        for i in range(self.nc):
            for j in range(self.ne):
                self.distanceXcXe[i][j] = np.abs((self.Xc[i] - self.Xe[j]))

    # notice that here we want to set PsicXeXc as ne * nc, this is an asmmetric matrix
    def distance_XeXc(self):
        self.distanceXeXc = np.zeros((self.ne,self.nc,self.k))
        for i in range(self.ne):
            for j in range(self.nc):
                self.distanceXeXc[i][j] = np.abs((self.Xe[i] - self.Xc[j]))

    # now let's calculate the matrix for Psic_Xc,Psic_Xe,Psic_XcXe, Psid_Xe. Cholesky decomposition is used
    def updatePsi(self):

        self.updateData()
        # note that there are 5 kinds of matrix in matrix C
        self.Psic_Xc = np.zeros((self.nc,self.nc), dtype=np.float)
        self.Psic_Xe = np.zeros((self.ne,self.ne), dtype=np.float)
        self.Psic_XcXe = np.zeros((self.nc,self.ne),dtype=np.float)
        self.Psic_XeXc = np.zeros((self.ne,self.ne),dtype=np.float)
        self.Psid_Xe = np.zeros((self.ne, self.ne),dtype=np.float)

        # now calculate Psi matrix
        # 1
        newPsic_Xc = np.exp(-np.sum(self.thetac*np.power(self.distanceXc,self.pc), axis = 2))
        self.Psic_Xc = np.triu(newPsic_Xc,1)
        self.Psic_Xc = self.Psic_Xc + self.Psic_Xc.T + np.mat(eye(self.nc)) + np.multiply(np.mat(eye(self.nc)),np.spacing(1))
        self.UPsic_Xc = np.linalg.cholesky(self.Psic_Xc)
        self.UPsic_Xc = self.UPsic_Xc.T  # the return value is a upper triangular matrix, L_T


        # 2
        newPsic_Xe = np.exp(-np.sum(self.thetac * np.power(self.distanceXe, self.pc), axis=2))
        self.Psic_Xe = np.triu(newPsic_Xe, 1)
        self.Psic_Xe = self.Psic_Xe + self.Psic_Xe.T + np.mat(eye(self.ne)) + np.multiply(np.mat(eye(self.ne)),np.spacing(1))
        self.UPsic_Xe = np.linalg.cholesky(self.Psic_Xe)
        self.UPsic_Xe = self.UPsic_Xe.T

        # 3
        self.Psic_XcXe = np.exp(-np.sum(self.thetac*np.power(self.distanceXcXe,self.pc),axis=2))

        # 4
        self.Psic_XeXc = np.exp(-np.sum(self.thetac * np.power(self.distanceXeXc,self.pc),axis=2))

        # 5
        self.Psid_Xe = np.exp(-np.sum(self.thetad * np.power(self.distanceXe, self.pd), axis=2))
        self.UPsid_Xe = np.linalg.cholesky(self.Psid_Xe)
        self.UPsid_Xe = self.UPsid_Xe.T


    def neglnlikelihood(self):

        ye = copy.deepcopy(self.ye)

        # calculate mu_d
        dd = ye - self.rho * self.yc_xe
        # a = np.linalg.solve(self.UPsid_Xe.T,dd)
        # b = np.linalg.solve(self.UPsid_Xe,a)
        c = ones([self.ne,1]).T.dot(np.mat(self.Psid_Xe).I.dot(dd))

        # d = np.linalg.solve(self.UPsid_Xe.T, ones([self.ne,1]))
        # e = np.linalg.solve(self.UPsid_Xe,d)
        f = ones([self.ne,1]).T.dot(np.mat(self.Psid_Xe).I.dot(ones([self.ne,1])))

        self.mud = c/f

        # let's solve sigmaSqrd
        # notice that here yc is alist, it should be transfered to n*1 array

        d = ye - self.rho * self.yc_xe - ones([self.ne, 1]).dot(self.mud)
        # a = np.linalg.solve(self.UPsid_Xe.T, d)
        # b = np.linalg.solve(self.UPsid_Xe, a)

        self.sigmaSqrd = d.T.dot(np.mat(self.Psid_Xe).I.dot(d)) / self.ne
        # self.LnDetPsid = 2. * np.sum(np.log(np.abs(np.diag(self.UPsid_Xe))))

        self.LnDetPsid = np.log(np.abs(np.linalg.det(self.Psid_Xe)))

        self.NegLnLike = -1.*(-(self.ne/2.)*np.log(self.sigmaSqrd[0,0]) - 0.5*self.LnDetPsid)


    def _getMatrixC(self):
        # note that here sigmaSqrc and sigmaSqrd are 1*1 matrix, it should be transfered to scalar
        C_up = np.hstack((self.sigmaSqrc*self.Psic_Xc,
                           self.rho*self.sigmaSqrc*self.Psic_XcXe))
        C_down = np.hstack((self.rho*self.sigmaSqrc*self.Psic_XeXc,
                           np.power(self.rho, 2) * self.sigmaSqrc * self.Psic_Xe + self.sigmaSqrd[0, 0] * self.Psid_Xe))

        self.C = np.vstack((C_up, C_down))
        self.UC = np.linalg.cholesky(self.C)

        # get mu
        yc = np.atleast_2d(self.yc).T  # here transfer the flat yc to vertical yc

        self.y = np.vstack((yc, self.ye))
        a = np.linalg.solve(self.UC, self.y)
        b = np.linalg.solve(self.UC.T, a)
        c = np.linalg.solve(self.UC, self.one)
        d = np.linalg.solve(self.UC.T, c)
        e = self.one.T.dot(b)
        f = self.one.T.dot(d)

        self.mu = e/f


    # ===========train low fidelity kriging model ======================
    def train_cheap(self,optimizer='pso'):
        self.cheapKriging = kriging(self.Xc, self.yc)
        self.cheapKriging.train(optimizer=optimizer)

        hyperpara_cheap = self.cheapKriging.get_theta_p()

        for i in range(self.k):
            self.thetac[i] = hyperpara_cheap[i]
        for i in range(self.k):
            self.pc[i] = hyperpara_cheap[i+self.k]

        print('thetac and pc is {}'.format(hyperpara_cheap))

        self.updateModel()

        musigma = self.cheapKriging.neglikelihood()
        self.muc = musigma[0]
        self.sigmaSqrc = musigma[1]
        self.LnDetPsid_cheap = musigma[2]
        self.NegLnLike_cheap = musigma[3]

        print('muc and sigmaSqrc are {}'.format([self.muc, self.sigmaSqrc]))
        print('likelihood_cheap = {}'.format(self.NegLnLike_cheap))


    # ===================== train the high fidelity kriging model ===================
    def train_expensive(self, optimizer='pso'):
        '''
        The function trains the hyperparameters of the expensive Kriging model.
        :param optimizer: Two optimizers are implemented, a Particle Swarm Optimizer or a GA
        '''

        # here we get the value of yc_xe
        for enu, entry in enumerate(self.Xe):
            if entry in self.Xc:
                # print('find this value {} in Xc!'.format(entry))
                index = np.argwhere(self.Xc == entry)[0,0]
                self.yc_xe.append(self.yc[index])
            else:
                print('find the value {} with kriging'.format(entry))
                y_predict = self.predict_cheap(entry)
                self.yc_xe.append(y_predict)

        self.yc_xe = np.atleast_2d(self.yc_xe).T  # transfer from list to np.array

        # then make sure our data is up-to-date
        self.updateData()

        # Establish the bounds for optimization for theta and p values
        lowerBound = [self.thetamin] * self.k + [self.pmin] * self.k + [self.rhomin]
        upperBound = [self.thetamax] * self.k + [self.pmax] * self.k + [self.rhomax]

        # Create a random seed for our optimizer to use
        rand = Random()
        rand.seed(int(time()))

        # If the optimizer option is PSO, run the PSO algorithm
        if optimizer == 'pso':
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            # ea.observer = inspyred.ec.observers.stats_observer
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=self.fittingObjective,
                                  pop_size=300,
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_evaluations=50000,
                                  neighborhood_size=30,
                                  num_inputs=self.k)
            # Sort and print the best individual, who will be at index 0.
            final_pop.sort(reverse=True)

        # If not using a PSO search, run the GA
        elif optimizer == 'ga2':
            ea = inspyred.ec.emo.NSGA2(Random())
            ea.terminator = self.no_improvement_termination

            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=self.fittingObjective,
                                  pop_size=10,
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_generations=50,
                                  num_elites=10,
                                  mutation_rate=0.1)

        # This code updates the model with the hyperparameters found in the global search
        for entry in final_pop:
            newValues = entry.candidate
            # newValues = [0.1, 0.1, 2, 2, 1]
            preLOP = copy.deepcopy(newValues)
            locOP_bounds = []
            for i in range(self.k):
                locOP_bounds.append([self.thetamin, self.thetamax])
            for i in range(self.k):
                locOP_bounds.append([self.pmin, self.pmax])
            locOP_bounds.append([self.rhomin,self.rhomax])
            # Let's quickly double check that we're at the optimal value by running a quick local optimizaiton
            lopResults = minimize(self.fittingObjective_local, newValues, method='SLSQP', bounds=locOP_bounds,
                                  options={'disp': False})
            # fun = lopResults['fun']

            newValues = lopResults['x']
        # Finally, set our new theta and pl values and update the model again
            for i in range(self.k):
                self.thetad[i] = newValues[i]
            for i in range(self.k):
                self.pd[i] = newValues[i + self.k]
            self.rho = newValues[self.k + self.k]
            try:
                self.updateModel()
            except:
                pass
            else:
                break

        print('succeed to train expensive kriging model')

        # set other paras
        # self.thetad = 0.1*np.ones(self.k)
        # self.rho = 1.25
        self.updateModel()
        self.neglnlikelihood()
        self._getMatrixC()
        print("mu = {}".format(self.mu))
        print("thetad, pd, rho is {}".format([self.thetad, self.pd, self.rho]))


    # ============================== optimization settings =============================
    def no_improvement_termination(self,population,num_generations,num_evaluations,args):
        '''
        return true if the best fitness does not change for a number of generations or the max number of
        evaluation is exceeded

        :param population: the population of individuals
        :param num_generations: the number of elapsed generations
        :param num_evaluations: the number of candidate solution evaluations
        :param args: a dictioary of keyword arguments
        '''
        max_generations = args.setdefault('max_generations',50)
        previous_best = args.setdefault('previous_best', None)
        max_evaluations = args.setdefault('max_evaluations', 50000)
        current_best = np.around(max(population).fitness, decimals=4)
        if previous_best is None or previous_best !=current_best:
            args['previous_best'] = current_best
            args['generation_count'] = 0
            return False or (num_evaluations >= max_evaluations)
        else:
            if args['generation_count'] >= max_generations:
                return True
            else:
                args['generation_count'] +=1
                return False or (num_evaluations >= max_evaluations)

    def generate_population(self,random,args):
        '''
        generate en initial population to train the expensive kriging
        :param args:args from the optimizer, like population
        '''
        size = args.get('num_inputs',None)
        bounder = args["_ec"].bounder
        chromosome = []
        for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
            chromosome.append(random.uniform(lo,hi))
        return chromosome

    def fittingObjective(self, candidates, args):
        '''
        The objective for a series of candidates from the hyperparameter global search.
        :param candidates: An array of candidate design vectors from the global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated NegLNLike values for the candidate population
        '''
        fitness = []
        for entry in candidates:
            f = 10000
            for i in range(self.k):
                self.thetad[i] = entry[i]
            for i in range(self.k):
                self.pd[i] = entry[i + self.k]
            self.rho = entry[self.k+self.k]

            # self.thetad = np.array([1, 1])
            # self.pd = np.array([2, 2])
            # self.rho = np.array([1.25])

            try:
                self.updateModel()
                self.neglnlikelihood()
                f = self.NegLnLike
                # print(f)
                # exit()
            except Exception as e:
                f = 10000
            fitness.append(f)
        return fitness

    def fittingObjective_local(self, entry):
        '''
        :param entry: The same objective function as the global optimizer, but formatted for the local optimizer
        :return: The fitness of the surface at the hyperparameters specified in entry
        '''
        f = 10000
        for i in range(self.k):
            self.thetad[i] = entry[i]
        for i in range(self.k):
            self.pd[i] = entry[i + self.k]
        self.rho = entry[self.k+self.k]
        try:
            self.updateModel()
            self.neglnlikelihood()
            f = self.NegLnLike
        except Exception as e:
            f = 10000
        return f

    # ====================  find the results, including prediction and error ========================
    def predict(self, x):
        x = copy.deepcopy(x)
        # get the distance between (Xc,x) and (Xe,x)
        distanceXcX = np.zeros((self.nc,1,self.k))
        for i in range(self.nc):
            distanceXcX[i,0] = np.abs((self.Xc[i]-x))
        distanceXeX = np.zeros((self.ne,1,self.k))

        for i in range(self.ne):
            distanceXeX[i,0] = np.abs((self.Xe[i] - x))

        Psic_x = np.exp(-np.sum(self.thetac*np.power(distanceXcX, self.pc), axis=2))
        Psid_x = np.exp(-np.sum(self.thetad*np.power(distanceXeX, self.pd), axis=2))
        Psic_Xex = np.exp(-np.sum(self.thetac*np.power(distanceXeX, self.pc), axis=2))

        # Psic_x = np.zeros([self.nc, 1])
        # for i in range(self.nc):
        #     Psic_x[i] = np.exp(-np.sum(self.thetac * np.power((np.abs(self.Xc[i] - x)), self.pc)))
        #
        # Psic_Xex = np.zeros([self.ne, 1])
        # for i in range(self.ne):
        #     Psic_Xex[i] = np.exp(-np.sum(self.thetac * np.power((np.abs(self.Xc[i] - x)), self.pc)))
        #
        # Psid_x = np.zeros([self.ne, 1])
        # for i in range(self.ne):
        #     Psid_x[i] = np.exp(-np.sum(self.thetad * np.power((np.abs(self.Xc[i] - x)), self.pd)))

        c1 = self.rho*self.sigmaSqrc*Psic_x
        c2 = np.power(self.rho, 2) * self.sigmaSqrc * Psic_Xex + self.sigmaSqrd[0, 0] * Psid_x
        c = np.vstack((c1, c2))
        a = np.linalg.solve(self.UC, self.y-self.one*self.mu)
        b = np.linalg.solve(self.UC.T, a)
        predict_value = self.mu + c.T.dot(b)
        predict_value = predict_value[0, 0]

        return predict_value

    def predicterr(self, x):
        x = copy.deepcopy(x)
        distanceXcX = np.zeros((self.nc, 1, self.k))
        for i in range(self.nc):
            distanceXcX[i, 0] = np.abs((self.Xc[i] - x))

        distanceXeX = np.zeros((self.ne, 1, self.k))
        for i in range(self.ne):
            distanceXeX[i, 0] = np.abs((self.Xe[i] - x))

        Psic_x = np.exp(-np.sum(self.thetac * np.power(distanceXcX, self.pc), axis=2))
        Psid_x = np.exp(-np.sum(self.thetad * np.power(distanceXeX, self.pd), axis=2))
        Psic_Xex = np.exp(-np.sum(self.thetac * np.power(distanceXeX, self.pc), axis=2))

        c1 = self.rho * self.sigmaSqrc * Psic_x
        c2 = np.power(self.rho, 2) * self.sigmaSqrc * Psic_Xex + self.sigmaSqrd[0, 0] * Psid_x
        c = np.vstack((c1, c2))
        a = np.linalg.solve(self.UC,c)
        b = np.linalg.solve(self.UC.T,a)

        error = np.power(self.rho, 2) * self.sigmaSqrc + self.sigmaSqrd - c.T.dot(b)
        error = np.sqrt(error[0, 0])

        return error

    # ===========find the cheap kriging prediction======================
    def predict_cheap(self, x):
        psi = np.zeros((self.nc, 1))
        one = np.ones(self.nc)
        for i in range(self.nc):
            psi[i] = np.exp(-np.sum(self.thetac*np.power((np.abs(self.Xc[i]-x)), self.pc)))
        z = self.yc-one.dot(self.muc)
        a = np.linalg.solve(self.UPsic_Xc.T, z)
        b=np.linalg.solve(self.UPsic_Xc, a)
        c=psi.T.dot(b)

        f=self.muc + c
        return f[0]

    def predicterr_cheap(self, x):
        SSqr=[]
        psi = np.zeros((self.nc, 1))
        one = np.ones(self.nc)
        for i in range(self.nc):
            try:
                psi[i] = np.exp(-np.sum(self.thetac*np.power((np.abs(self.Xc[i]-x)), self.pc)))
            except Exception, e:
                print Exception, e
        try:
            SSqr = self.sigmaSqrc*(1-psi.T.dot(np.linalg.solve(self.UPsic_Xc, np.linalg.solve(self.UPsic_Xc.T, psi))))
        except Exception, e:
            print Exception, e
            pass

        SSqr = np.abs(SSqr[0])
        return np.power(SSqr, 0.5)[0]

    # =====================cheap kriging prediction end ========================


    # ==================== infill method =====================================
    # here we add a function to add points with two different methods
    def infill(self,points,method='error',optimizer='PSO'):

        returnValues = np.zeros([points,self.k],dtype=float)
        for i in range(points):
            rand = Random()
            rand.seed(int(time()))
            if optimizer == 'PSO':
                ea = inspyred.swarm.PSO(Random())  # use PSO algorithms
                ea.terminator = self.no_improvement_termination
                ea.topology = inspyred.swarm.topologies.ring_topology

                if method == 'ei':
                    evaluator = self.infill_objective_ei
                else:
                    evaluator = self.infill_objective_mse
 
                final_pop = ea.evolve(generator=self.generate_population,
                                    evaluator=evaluator,
                                    pop_size=200,
                                    maximize=False,
                                    bounder=ec.Bounder([0.] * self.k, [1.]*self.k),
                                    max_evaluations=20000,
                                    neighborhood_size=40,
                                    num_inputs=self.k)
                final_pop.sort(reverse=True)
            
            elif optimizer == 'ga2':
                ea = inspyred.ec.emo.NSGA2(Random())
                ea.terminator = self.no_improvement_termination
                if method == 'ei':
                    evaluator = self.infill_objective_ei
                else:
                    evaluator = self.infill_objective_mse

                final_pop = ea.evolve(generator=self.generate_population,
                                    evaluator=evaluator,
                                    pop_size=100,
                                    maximize=False,
                                    bounder=ec.Bounder([0.] * self.k, [1.]*self.k),
                                    max_generations=50,
                                    num_elites=10,
                                    mutation_rate=0.1)
                final_pop.sort(reverse=True)

            newpoint = final_pop[0].candidate
            returnValues[i][:] = newpoint

        return returnValues

    # here is the measured value of infill method, two measuring methods
    # the value of expected improvement EI
    def expimp(self,x):
        EI = 0
        S = self.predicterr(x)

        y_min = np.min(self.yc)
        if S<=0.:
            EI = 0.
        elif S>0.:
            EI_one = (y_min - self.predict(x)) * norm.cdf((y_min - self.predict(x))/ S)
                
            EI_two = S * norm.pdf((y_min - self.predict(x))/S)
            EI = EI_one +EI_two

        return EI

    def infill_objective_ei(self,candidates, args):
        fitness = []
        for entry in candidates:
            fitness.append(-1*self.expimp(entry))
        return fitness

    def infill_objective_mse(self,candidates, args):
        fitness = []
        for entry in candidates:
            fitness.append(-1*self.predicterr(entry))
        return fitness

    # add point to the input list, and update the model
    def addPoint(self, newX, newy):

        self.Xe = np.append(self.Xe, [newX], axis=0)
        self.ye = np.append(self.ye, [newy], axis=0)
        self.ne = self.Xe.shape[0]
        self.yc_xe = []
        self.updateData()
        self.updateModel()
        self.train_expensive()


    # =======================infill method end======================

    # calculate the mean error
    def calculate_mean_error(self, points_num=100):
        sp = samplingplan(k=2)
        points = sp.rlh(points_num)
        err = np.zeros(points_num)
        for i in range(points_num):
            err[i] = self.predicterr(points[i])
        return np.mean(err)


    def get_xc(self):
        return self.Xc

    def get_yc(self):
        return self.yc
