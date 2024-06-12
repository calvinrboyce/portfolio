import networkx as nx
import matplotlib.pyplot as plt
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    def __init__(self, filename="data/movie_data.txt", n_movies=-1):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        #sets and graph
        self.titles = set()
        self.actors = set()
        self.g = nx.Graph()
        
        #read file and fill sets and graph
        with open(filename) as file:
            movies = file.readlines()
        for movie in movies[:n_movies]:
            movie = movie[:-1]
            contents = movie.split('/')
            self.titles.add(contents[0])
            for actor in contents[1:]:
                self.actors.add(actor)
                self.g.add_edge(contents[0], actor)
                
                
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        #find and return the shortest path
        path = nx.shortest_path(self.g, source, target)
        return len(path)//2, path

    
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        #calculate paths and lengths
        paths = nx.shortest_path_length(self.g, target)
        lengths = []
        for actor in paths.keys():
            if actor in self.actors:
                lengths.append(paths[actor]//2)
        
        #plot
        plt.hist(lengths, bins=[i-.5 for i in range(8)])
        plt.title('Path Lengths between ' + target + ' and All Other Actors')
        plt.xlabel('Path Lengths')
        plt.ylabel('Number of Actors')
        plt.show()
        
        return sum(lengths)/len(lengths)