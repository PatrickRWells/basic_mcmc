import numpy as np
import matplotlib.pyplot as plt

class Posterior:
    def __init__(self, num=100):

        """
        A posterior distribution. Basically a fake likelihood that draws from
        a gaussian with unit variance

        Attributes
        ----------
        samples: list of floats
            random samples from which the posterior is constructed
        num: the number of samples (default 100)

        total:
            0.5 times the sum of the squares of the samples
            used when finding probabilities

        """


        self.samples = np.random.normal(size=num)
        self.num = num
        self.total = 0
        self.total += np.sum(self.samples**2)
        
        self.total = self.total/2

    def prob(self, a):

        """
        Returns the posterior probabiliity for a given value of a

        Parameters
        -----------
        
        a: float
            the value of the parameter
        """

        fact = (np.log(a)/(2*np.pi))**(self.num/2)
        exp = a**(-self.total)
        return fact*exp

    def plot_distribution(self, amin, amax, filename='posterior.png'):

        """
        Plots the posterior distribution
        
        Parameters
        -----------
        amin: float
            minimum value to plot
        amax: float
            maximum value to plat
        location: string
            location where the figure will be saved
            defualts to home folder
        """

        a = np.linspace(amin, amax, 1000)
        ys = self.prob(a)
        plt.plot(a,ys)
        plt.savefig(filename)
        plt.close()

class MCMC:

    def __init__(self):

        """
        A basic MCMC sampler, samples over a gausian distribution
        with unit variance
        
        Attributes
        -----------
        posterior: Posterior object
            The "fake" posterior
        
        variance:
            The variance of the generating function
            this should be set when running the mcmc 
        """

        

        self.posterior = Posterior()
        self.variance = 0
    
    def plot_posterior(self, amin = 1, amax = 5, filename='posterior.png'):
        """
        Plots the posterior distribution
        
        Parameters
        -----------
        amin: float
            minimum value to plot
        amax: float
            maximum value to plat
        location: string
            location where the figure will be saved
            defualts to home folder
        """
        self.posterior.plot_distribution(amin, amax, filename)

    
    def draw_sample(self, xt):

        """
        Draws a new location for the MCMC
        from a gaussian distribution
        
        Parameters
        ----------
        xt: float
            center of the distribution
        
        """

        return np.random.normal(xt, self.variance)

    def walk_probability(self, xp, xt):
        """
        Returns the value of the generating function
       for a given walk

        
        Parameters
        ----------
        xt: float
            Initial point
        xp: float
            New point

        """
        exponent = -((xp - xt)**2)/(2*(self.variance**2))
        fact = 1/(np.sqrt(2*np.pi)*self.variance)
        return fact*np.exp(exponent)

    def check_acceptance(self, prob):
        """
        Determines if a step will be accepted
        based on its probability
        
        Parameters
        ----------
        prob: float
            probability of step, determined by mcmc
        """
    
        if prob > 1:
            return True
        else:
            return True if prob >= np.random.uniform() else False

    def run_mcmc(self, steps = 100000, initial = 2.7, variance = 0.05):
        """
        Runs an MCMC and returns the chain
        Default values have been set for this homework assignment
        
        Parameters
        ----------
        steps: int
            number of steps to take
        initial: float
            starting location of the mcmc
        variance: float
            variance of the generating function

        Returns
        -------
        chain: Chain object
        """
        self.variance = variance
        chain = Chain(initial)
        xt = initial

        for _ in range(steps):
            xp = self.draw_sample(xt)
            if xp <= 1:
                chain.add_sample(xt)
                continue
            
            prob = self.posterior.prob(xp)/self.posterior.prob(xt)
            if self.check_acceptance(prob):
                chain.add_sample(xp)
                xt = xp
            else:
                chain.add_sample(xt)
    
        return chain

class Chain:
    def __init__(self, x0):
        """
        Object that stores MCMC samples

        Attributes:
        -----------
        samples: list of floats:
            List containing the links in the chain
            (for a single parameters)
        """
        self.samples = [x0]
    
    def add_sample(self, x):
        
        """
        Adds a sample to the chain

        Parameters:
        -----------
        x: float:
            value of the next link in the chain
        """

        self.samples = np.append(self.samples, x)
    
    def plot_histogram(self, bins = 100, filename='histogram.png'):
        """
        Plots a histogram for sampels in the chain

        Parameters:
        -----------
        bins: int
            number of bins in the histogram
        
        location: string
            location to save the figure
        """
        plt.hist(self.samples, bins)
        plt.savefig(filename)
        plt.close()
    
    def trace_plot(self, filename='trace_plot.png'):
        """
        Plots the value of the parameter as a function
        of sample number

        Parameters
        -----------
        location: string
            File location to save the figure
        """
        nums = [i+1 for i in range(len(self.samples))]
        plt.plot(nums, self.samples)
        plt.savefig(filename)
        plt.close()




if __name__ == "__main__":

    mcmc = MCMC()
    mcmc.plot_posterior()
    chain = mcmc.run_mcmc()
    chain.plot_histogram()
    chain.trace_plot()
    #Note that by default the figures will be saved to your home directory





