import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        A (np.ndarray): an nxn transition matrix
        states (list): a list of state labels
        indices (dict): a maps the state labels to the row/column 
                index they correspond to
    """
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        #check if it's stochastic
        if not np.allclose(A.sum(axis=0),np.ones(A.shape[0])):
            raise ValueError('A is not stochastic')
        if A.shape[0] != A.shape[1]:
            raise ValueError('A is not square')
        self.A = A
            
        #fill dictionary
        if states is None:
            states = [i for i in range(A.shape[0])]
        self.states = states
        self.indices = dict()
        for i, label in enumerate(self.states):
            self.indices[label] = i
        

    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        #draw
        draw = np.random.multinomial(1, self.A[:,self.indices[state]])
        
        return list(self.indices.keys())[np.argmax(draw)]

    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        #store labels at each step
        walk = [start]
        for i in range(N-1):
            start = self.transition(start)
            walk.append(start)
        
        return walk

    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        #transition until start = stop
        path = [start]
        while start != stop:
            start = self.transition(start)
            path.append(start)
            
        return path

    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        #starting state
        state = np.random.random(len(self.states))
        state = state/sum(state)
        state1 = self.A@state
        iters = 0
        
        #iterate until convergence or tolerance
        while np.linalg.norm(state - state1, ord=1) > tol:
            if iters > maxiter:
                raise ValueError("A^k does not converge")
            state, state1 = state1, self.A@state1
            iters +=1
            
        return state1


class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        A (np.ndarray): an nxn transition matrix
        states (list): a list of state labels
        indices (dict): a maps the state labels to the row/column
                index they correspond to
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        #read file
        with open(filename) as file:
            file = file.read()
        
        #get words
        wordlist = list(set(file.split()))
        wordlist.append('$tart')
        wordlist.append('$top')
        
        #transition matrix
        n = len(wordlist)
        A = np.zeros((n,n))
        for sentence in file.split('\n'):
            words = sentence.split()
            words.insert(0,'$tart')
            words.append('$top')
            for x, y in zip(words[:-1], words[1:]):
                A[wordlist.index(y),wordlist.index(x)] += 1
        
        #normalize
        A[n-1,n-1] = 1
        sum = A.sum(axis=0)
        for i in range(n):
            for j in range(n):
                A[i,j]/=sum[j]
        
        #save attributes
        self.A = A
        self.states = wordlist
        self.indices = dict()
        for i, label in enumerate(self.states):
            self.indices[label] = i
        

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        #use path to generate a sentence
        sentence = self.path('$tart', '$top')
        return ' '.join(sentence[1:-1])
        
