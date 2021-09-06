import copy
import itertools
import unittest



class GraphSample:
    def __init__(self, markers, connections):
        self.markers = markers
        self.connections = connections


#
# Interaction matrix is symmetric, so for save memory the represented as lower
# triangular matrix. For example graph with 4 cliques is like:
#
#     +-+
#  0  | |
#     +-+-+
#  1  |0| |
#     +-+-+-+
#  2  |1|2| |
#     +-+-+-+-+
#  3  |3|4|5| |
#     +-+-+-+-+
#      0 1 2 3
#
# All elements in main diagonal elemnts are 0 so doesn't need to save them.
# Number of elements of graph with 'n' cliques is:
#   num_elem = n * (n + 1) / 2 - n
#
# Index of (i, j) element, where j is greater than i, is:
#    idx = j * (j - 1) / 2 + i
#
class Graph:
    def __init__(self):
        self.cliques = []                                                       # Markers in cliques
        self.delta = []                                                         # Interactions


    def GetNumCliques(self):
        return len(self.cliques)


    def GetNumMarkers(self, clique_idx):
        if clique_idx < 0 or clique_idx >= self.GetNumCliques():
            return None

        return len(self.cliques[clique_idx])


    def GetTotalNumMarkers(self):
        total_num_markers = 0
        for c in self.cliques:
            total_num_markers += len(c)
        return total_num_markers


    def GetClique(self, clique_idx):
        if clique_idx < 0 or clique_idx >= self.GetNumCliques():
            return []
        return self.cliques[clique_idx]


    def GetMarkers(self, clique_idx):
        return self.GetClique(clique_idx)


    def GetCliqueCopy(self, clique_idx):
        if clique_idx < 0 or clique_idx >= self.GetNumCliques():
            return []
        return copy.deepcopy(self.cliques[clique_idx])


    def GetMarkersCopy(self, clique_idx):
        return self.GetCliqueCopy(clique_idx)


    def GetConnections(self, clique_idx):
        conns = [ x for x in range(self.GetNumCliques())
                    if x != clique_idx and self.HasInteraction(clique_idx, x) ]
        return conns


    def AddClique(self, clique):
        self.cliques.append(copy.deepcopy(clique))                              # Deep copy markers.
        if self.GetNumCliques() > 1:
            n = self.GetNumCliques() - 1
            self.delta.extend(itertools.repeat(0, n))                           # Append 0's to the end of interactions.


    def RemoveClique(self, i):
        if self.GetNumCliques() == 0 or i >= self.GetNumCliques() or i < 0:
            return False

        if self.GetNumCliques() == 1:
            del self.cliques[i]
            return True

        # Remove from interaction list.
        off = 0
        for row in range(i + 1, self.GetNumCliques()):
            idx = self.GetIdx(row, i) - off
            del self.delta[idx]                     # Remove columns.
            off += 1
        idx = self.GetIdx(i, 0)
        del self.delta[idx : idx + i]               # Remove row.

        # Remove from clique list.
        del self.cliques[i]
        return True


    def GetIdx(self, i, j):
        if i > j:
            tmp = i; i = j; j = tmp     # Swap i and j.
        idx = j * (j - 1) / 2 + i
        return idx


    def SetInteraction(self, i, j, val=1):
        num_elems = self.GetNumCliques()
        if num_elems < 2 or i >= num_elems or j >= num_elems or i == j:
            return False

        self.delta[self.GetIdx(i, j)] = val
        return True


    def HasInteraction(self, i, j):
        return 0 if i == j or self.GetNumCliques() < 1 else self.delta[self.GetIdx(i, j)]


    def FindMarker(self, marker):
        for i in range(self.GetNumCliques()):
            if marker in self.cliques[i]:
                return i, self.cliques[i].index(marker)
        return None, None


    def AddMarker(self, clique_idx, marker):
        if clique_idx < 0 or clique_idx >= self.GetNumCliques():
            return False

        if marker not in self.cliques[clique_idx]:
            self.cliques[clique_idx].append(marker)
        return True


    def RemoveMarker(self, marker):
        clique_idx, marker_idx = self.FindMarker(marker)
        return self.RemoveMarkerFromClique(clique_idx, marker_idx)


    def RemoveMarkerFromClique(self, clique_idx, marker_idx):
        if clique_idx == None or marker_idx == None:
            return False
        if clique_idx < 0 or clique_idx >= self.GetNumCliques():
            return False
        if marker_idx < 0 or marker_idx >= len(self.cliques[clique_idx]):
            return False

        del self.cliques[clique_idx][marker_idx]
        if len(self.cliques[clique_idx]) == 0:
            self.RemoveClique(clique_idx)
        return True


    def RemoveInteractions(self, clique_idx):
        if clique_idx < 0 or clique_idx >= self.GetNumCliques():
            return False

        for i in range(self.GetNumCliques()):
            if self.HasInteraction(clique_idx, i):
                self.SetInteraction(i, clique_idx, 0)
                self.SetInteraction(clique_idx, i, 0)
        return True


    def GetComponents(self):
        if self.GetNumCliques() < 1:
            return []
        if self.GetNumCliques() == 1:
            return [[0]]

        components = []
        visited_nodes = [ False for i in range(self.GetNumCliques()) ]
        for i in range(self.GetNumCliques()):
            if visited_nodes[i] == True:
                continue

            components.append([])                                               # Add new component.
            node_stack = [ i ]
            j = 0
            while j < len(node_stack):
                n = node_stack[0]                                               # Get element from node stack.
                del node_stack[0]
                if visited_nodes[n]:
                    continue

                components[-1].append(n)
                visited_nodes[n] = True
                for k in range(self.GetNumCliques()):                           # Visit adjacent nodes.
                    if visited_nodes[k] == False and self.HasInteraction(n, k):
                        visited_nodes[k] = True
                        components[-1].append(k)
                        node_stack.append(k)
                j += 1
        return components


    def ToString(self):
        num_elems = self.GetNumCliques()
        s = 'Cliques({}):\n'.format(num_elems)
        for i in range(num_elems):
            s += ' ' + str(self.cliques[i]) + '\n'

        s += '\nInteractions({}):\n'.format(len(self.delta))
        s += ' ' + str(self.delta) + '\n\n'
        for i in range(1, num_elems):
            s += ' '
            for j in range(i):
                s += '1' if self.HasInteraction(i, j) else '0'
            if i + 1 < num_elems:
                s += '\n'
        return s


    def GetGraphSample(self):
        samples = []
        for c in range(self.GetNumCliques()):
            smpl = GraphSample(self.GetCliqueCopy(c), self.GetConnections(c))
            samples.append(smpl)
        return samples


    def IsOverlapping(self):
        for x in range(self.GetNumCliques()):
            for y in range(self.GetNumCliques()):
                if x == y:
                    continue
                for m in self.GetMarkers(x):
                    if m in self.GetMarkers(y):
                        return True
        return False



class DiseaseGraphTestCase(unittest.TestCase):
    def test_DiseaseGraph1(self):
        G = Graph()
        G.AddClique([0, 1, 2])
        G.AddClique([5, 6])
        G.AddClique([7, 8, 9])
        G.AddClique([11, 15, 20, 23])
        G.AddClique([24, 25])

        self.assertEqual((0, 1), G.FindMarker(1))
        self.assertEqual((1, 0), G.FindMarker(5))
        self.assertEqual((2, 2), G.FindMarker(9))
        self.assertEqual((3, 3), G.FindMarker(23))
        self.assertEqual((4, 1), G.FindMarker(25))
        self.assertEqual((None, None), G.FindMarker(26))

        self.assertTrue(G.SetInteraction(2, 1))
        self.assertTrue(G.SetInteraction(3, 0))
        self.assertTrue(G.SetInteraction(3, 2))
        self.assertTrue(G.SetInteraction(4, 0))
        self.assertTrue(G.SetInteraction(4, 2))
        self.assertTrue(G.SetInteraction(4, 3))

        self.assertFalse(G.HasInteraction(1, 0))
        self.assertFalse(G.HasInteraction(2, 0))
        self.assertTrue(G.HasInteraction(2, 1))
        self.assertTrue(G.HasInteraction(3, 0))
        self.assertFalse(G.HasInteraction(3, 1))
        self.assertTrue(G.HasInteraction(3, 2))
        self.assertTrue(G.HasInteraction(4, 0))
        self.assertFalse(G.HasInteraction(4, 1))
        self.assertTrue(G.HasInteraction(4, 2))
        self.assertTrue(G.HasInteraction(4, 3))
        print(G.ToString() + '\n')

        self.assertTrue(G.RemoveClique(2))
        self.assertFalse(G.HasInteraction(1, 0))
        self.assertTrue(G.HasInteraction(2, 0))
        self.assertFalse(G.HasInteraction(2, 1))
        self.assertTrue(G.HasInteraction(3, 0))
        self.assertFalse(G.HasInteraction(3, 1))
        self.assertTrue(G.HasInteraction(3, 2))
        self.assertEqual((0, 1), G.FindMarker(1))
        self.assertEqual((1, 0), G.FindMarker(5))
        self.assertEqual((None, None), G.FindMarker(9))
        self.assertEqual((2, 3), G.FindMarker(23))
        self.assertEqual((3, 1), G.FindMarker(25))
        self.assertEqual((None, None), G.FindMarker(26))
        print(G.ToString() + '\n')

        self.assertTrue(G.RemoveClique(0))
        self.assertFalse(G.HasInteraction(1, 0))
        self.assertFalse(G.HasInteraction(2, 0))
        self.assertTrue(G.HasInteraction(2, 1))
        self.assertEqual((None, None), G.FindMarker(1))
        self.assertEqual((0, 0), G.FindMarker(5))
        self.assertEqual((None, None), G.FindMarker(9))
        self.assertEqual((1, 3), G.FindMarker(23))
        self.assertEqual((2, 1), G.FindMarker(25))
        self.assertEqual((None, None), G.FindMarker(26))
        print(G.ToString() + '\n')


    #
    # Graph structure (5 cliques, an and 2 components):
    #
    #         .------.       .------.
    #        /   n0   \____ /   n1   \
    #        \ {0, 1} /     \ {2, 3} /
    #         '------'       '------'
    #
    #      .---------.       .--------.
    #     /     n2    \____ /    n3    \
    #     \ {5, 6, 7} /     \ {10, 13} /
    #      '---------'       '--------'
    #              \
    #               \ .----.
    #                /  n4  \
    #                \ {33} /
    #                 '----'
    #
    # Connected components (modules):
    #     { n0, n1 }
    #     { n2, n3, n4 }
    #
    def test_DiseaseGraph2(self):
        G = Graph()
        G.AddClique([0, 1])
        G.AddClique([2, 3])
        G.AddClique([5, 6, 7])
        G.AddClique([10, 13])
        G.AddClique([33])
        self.assertTrue(G.SetInteraction(0, 1))
        self.assertTrue(G.SetInteraction(2, 3))
        self.assertTrue(G.SetInteraction(2, 4))
        components = G.GetComponents()
        self.assertEqual(2, len(components))                                                 # Check number of components.
        self.assertEqual([ [0, 1], [2, 3, 4] ], components)                                  # Check components themselves.


    def test_RemoveInteractions1(self):
        #
        #   [0] -- [1] -- [2]
        #      __________  |
        #     /          \ |
        #   [5] -- [4] -- [3]
        #
        G = Graph()
        G.AddClique([0])
        G.AddClique([1])
        G.AddClique([2])
        G.AddClique([3])
        G.AddClique([4])
        G.AddClique([5])
        self.assertTrue(G.SetInteraction(0, 1))
        self.assertTrue(G.SetInteraction(1, 2))
        self.assertTrue(G.SetInteraction(2, 3))
        self.assertTrue(G.SetInteraction(3, 4))
        self.assertTrue(G.SetInteraction(3, 5))
        self.assertTrue(G.SetInteraction(4, 5))
        print(G.ToString())
        self.assertTrue(G.RemoveInteractions(3))
        self.assertFalse(G.HasInteraction(3, 0))
        self.assertFalse(G.HasInteraction(3, 1))
        self.assertFalse(G.HasInteraction(3, 2))
        self.assertFalse(G.HasInteraction(3, 3))
        self.assertFalse(G.HasInteraction(3, 4))
        self.assertFalse(G.HasInteraction(3, 5))
        self.assertTrue(G.HasInteraction(0, 1))
        self.assertTrue(G.HasInteraction(1, 2))
        self.assertTrue(G.HasInteraction(4, 5))
        print(G.ToString())


    def test_RemoveInteractions2(self):
        #
        #        [1]
        #       / | \
        #      /  |  \
        #   [0]--[2]--[3]
        #
        G = Graph()
        G.AddClique([0])
        G.AddClique([1])
        G.AddClique([2])
        G.AddClique([3])
        self.assertTrue(G.SetInteraction(0, 1))
        self.assertTrue(G.SetInteraction(0, 2))
        self.assertTrue(G.SetInteraction(1, 2))
        self.assertTrue(G.SetInteraction(1, 3))
        self.assertTrue(G.SetInteraction(2, 3))
        self.assertFalse(G.HasInteraction(0, 3))
        print(G.ToString())
        self.assertTrue(G.RemoveInteractions(2))
        self.assertFalse(G.HasInteraction(0, 2))
        self.assertFalse(G.HasInteraction(1, 2))
        self.assertFalse(G.HasInteraction(3, 2))
        self.assertFalse(G.HasInteraction(0, 3))
        self.assertTrue(G.HasInteraction(0, 1))
        self.assertTrue(G.HasInteraction(1, 3))
        print(G.ToString())


    def test_GetConnections(self):
        #
        #        [1]
        #       / | \
        #      /  |  \
        #   [0]--[2]--[3]
        #
        G = Graph()
        G.AddClique([0])
        G.AddClique([1])
        G.AddClique([2])
        G.AddClique([3])
        self.assertTrue(G.SetInteraction(0, 1))
        self.assertTrue(G.SetInteraction(0, 2))
        self.assertTrue(G.SetInteraction(1, 2))
        self.assertTrue(G.SetInteraction(1, 3))
        self.assertTrue(G.SetInteraction(2, 3))
        self.assertFalse(G.HasInteraction(0, 3))
        self.assertEqual([1, 2], G.GetConnections(0))
        self.assertEqual([0, 2, 3], G.GetConnections(1))
        self.assertEqual([0, 1, 3], G.GetConnections(2))
        self.assertEqual([1, 2], G.GetConnections(3))


    def test_GetGraphSample(self):
        #
        #        [1]
        #       / | \
        #      /  |  \
        #   [0]--[2]--[3]
        #
        G = Graph()
        G.AddClique([0])
        G.AddClique([1])
        G.AddClique([2])
        G.AddClique([3])
        self.assertTrue(G.SetInteraction(0, 1))
        self.assertTrue(G.SetInteraction(0, 2))
        self.assertTrue(G.SetInteraction(1, 2))
        self.assertTrue(G.SetInteraction(1, 3))
        self.assertTrue(G.SetInteraction(2, 3))
        self.assertFalse(G.HasInteraction(0, 3))
        graph_sample = G.GetGraphSample()
        self.assertEqual(4, len(graph_sample))
        self.assertEqual([0],       graph_sample[0].markers)
        self.assertEqual([1, 2],    graph_sample[0].connections)
        self.assertEqual([1],       graph_sample[1].markers)
        self.assertEqual([0, 2, 3], graph_sample[1].connections)
        self.assertEqual([2],       graph_sample[2].markers)
        self.assertEqual([0, 1, 3], graph_sample[2].connections)
        self.assertEqual([3],       graph_sample[3].markers)
        self.assertEqual([1, 2],    graph_sample[3].connections)


    def test_RemoveMarkerFromClique(self):
        #
        #  [0, 1, 2] --- [3, 4]
        #      |
        #      |
        #     [5]
        #
        G = Graph()
        G.AddClique([0, 1, 2])
        G.AddClique([3, 4])
        G.AddClique([5])
        self.assertEqual(3, G.GetNumCliques())
        self.assertTrue(G.SetInteraction(0, 1))
        self.assertTrue(G.SetInteraction(0, 2))
        print(G.ToString())
        self.assertTrue(G.RemoveMarkerFromClique(2, 0))
        self.assertEqual(2, G.GetNumCliques())



if __name__ == '__main__':
    unittest.main()

