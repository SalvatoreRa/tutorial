### INTRODUCTION TO THE USE OF IGRAPH ###
#this is just a series of examples you can use or try
#use and import in R studio to try



### BASIC OPERATION ###

## Load Igraph package
library(igraph)

#make a ring graph
g <- make_ring(10)

#to see the edges of the graph
print_all(g)

#undirected graph
g <- graph( edges=c(1,2, 2,3, 3, 1), n=3, directed=F ) 
plot(g)

#directed graph
g <- graph( edges=c(1,2, 2,3, 3, 1), n=3, directed=T ) 
plot(g)

#class of the object
class(g)

#print the graph
g


#named vertex
g <- graph( c("Bob", "Joe", "Joe", "Jeff", "Jeff", "Bob")) 
#we do not need the number of the nodes, it infer by itself
plot(g)
g

g <- graph( c("Bob", "Joe", "Joe", "Jeff", "Jeff", "Bob"),
            isolates = c("carl", "louis"))
plot(g)

#a more complex plotting
plot(g, edge.arrow.size=.5, 
     vertex.color="green", #color of vertex
     vertex.size=15, #size vertex
     vertex.frame.color="black", #border vertex color
     vertex.label.color="red", #vertex label
     vertex.label.cex=0.8, 
     vertex.label.dist=2, 
     edge.curved=0.2)  #edge are curved

#alternative system
#notice that here the grammar is different
# - undirected graph
# +- for directed source
# -+ for directed target
# ++ symmetric relationship
# : set of vertices

plot(graph_from_literal(a---b, b---c))
plot(graph_from_literal(a--+b, b+--c))
plot(graph_from_literal(a+-+b, b+-+c)) 
plot(graph_from_literal(a:b:c---c:d:e))
g <- graph_from_literal(a-b-c-d-e-f, a-g-h-b, h-e:f:i, j)
plot(g)

# attributes for edges and nodes
#the edges
#you can print all the edges or manipulate them
E(g)
#vertex
V(g)
#get adiacency matrix
g[]
#you can subset the matrix
g[2,]

#the name attribute is automatically generated when you
# create the graph
V(g)$name

#assigning new attributes
#we are assigning a new attribute to the node
#arbitrary attribute named tyep_vertex
V(g)$type_vertex<-c("bla", "bla bla", "bla", "bla bla", "bla", "bla bla",
                    "bla", "bla bla", "bla", "bla bla")


#we are assigning the same attribute to all the edges here
E(g)$weight <- 8

#attributes of the graoh
graph_attr(g)

#set attributes 
g <- set_graph_attr(g, "something", "bla bla bla")
g <- set_edge_attr(g, "something edge", "bla bla bla")
g <- set_vertex_attr(g, "something vertex", "bla bla bla")

#delete attribute
g <- delete_graph_attr(g, "something")


#other type of graph
#Empty graph
g <- make_empty_graph(20)
plot(g, vertex.size=10, vertex.label=NA)

#full graph
g <- make_full_graph(20)
plot(g, vertex.size=10, vertex.label=NA)

#star graph
g <- make_star(20)
plot(g, vertex.size=10, vertex.label=NA)


#a tree
g <- make_tree(40, children = 3, mode = "undirected")
plot(g, vertex.size=10, vertex.label=NA)

g <- make_tree(40, children = 3, mode = "out")
plot(g, vertex.size=10, vertex.label=NA)

g <- make_tree(40, children = 3, mode = "in")
plot(g, vertex.size=10, vertex.label=NA)


#ring graph
g <- make_ring(40)
plot(g, vertex.size=10, vertex.label=NA)


#Erdos-Renyi random graph model
#n number of nodes
#m number of edges
g <- sample_gnm(n=100, m=60) 

plot(g, vertex.size=6, vertex.label=NA)  

#Watts-Strogatz small-world model
g <- sample_smallworld(dim=2, size=10, nei=1, p=0.2)
plot(g, vertex.size=6, vertex.label=NA, 
     layout=layout_in_circle)


#Barabasi-Albert preferential attachment model for scale-free graphs
g <-  sample_pa(n=100, power=1, m=1,  directed=F)
plot(g, vertex.size=6, vertex.label=NA)

# the Zachary carate club
#historical graph
g <- graph("Zachary") 
plot(g, vertex.size=10, vertex.label=NA)



#plot layaout
g <- sample_pa(80) 
V(g)$size <- 8
V(g)$frame.color <- "white"
V(g)$color <- "orange"
V(g)$label <- "" 
E(g)$arrow.mode <- 0
plot(g)

#layout random
plot(g, layout=layout_randomly)

#layout in circle
plot(g, layout=layout_in_circle)

#layout in sphere
plot(g, layout=layout_on_sphere)


# Fruchterman Reingold
plot(g, layout=layout_with_fr)

#Kamada Kawai
plot(g, layout=layout_with_kk)

#LGL algorithm
plot(g, layout=layout_with_lgl)



#plotting more than one graph
par(mfrow=c(2,2), mar=c(0,0,0,0))  

plot(g, layout=layout_with_fr)
plot(g, layout=layout_with_fr)
plot(g, layout=layout_on_sphere)
plot(g, layout=layout_on_sphere)

dev.off()

#rescaling
l <- layout_with_fr(g)

l <- norm_coords(l, ymin=-1, ymax=1, xmin=-1, xmax=1)

par(mfrow=c(2,2), mar=c(0,0,0,0))
plot(g, rescale=F, layout=l*0.4)
plot(g, rescale=F, layout=l*0.6)
plot(g, rescale=F, layout=l*0.8)
plot(g, rescale=F, layout=l*1.0)

dev.off()


## visualize all the layouts
#all the available layout
layouts <- grep("^layout_", ls("package:igraph"), value=TRUE)[-1] 
par(mfrow=c(3,3), mar=c(1,1,1,1))


layouts <- layouts[!grepl("bipartite|merge|norm|sugiyama|tree", layouts)]

for (layout in layouts) {
  
  print(layout)
  
  l <- do.call(layout, list(g)) 
  
  plot(g, edge.arrow.mode=0, layout=l, main=layout) }

dev.off()


#graph properties
#proportion of all the edges over the possible ones
edge_density(g, loops=F)

#transitivity
transitivity(g, type="global")

#diameter
diameter(g, directed=F, weights=NA)

#node degree
deg <- degree(g, mode="all")
plot(g, vertex.size=deg*3)

hist(deg, breaks=1:vcount(g)-1, 
     main="node degree")

#degree distribution
deg.dist <- degree_distribution(g, cumulative=T, mode="all")

plot( x=0:max(deg), y=1-deg.dist, pch=19, cex=1.2, 
      col="orange", 
      xlab="Degree", ylab="Cumulative Frequency")


#centrality measures

#degree centrality
degree(g, mode="in")

centr_degree(g, mode="in", normalized=T)

#closeness centrality
closeness(g, mode="all", weights=NA) 

centr_clo(g, mode="all", normalized=T) 


#eigen vectore centrality
eigen_centrality(g, directed=T, weights=NA)

centr_eigen(g, directed=T, normalized=T) 


#betweeneness centrality
betweenness(g, directed=T, weights=NA)

edge_betweenness(g, directed=T, weights=NA)

centr_betw(g, directed=T, normalized=T)


#distance and path
mean_distance(g, directed=F)
mean_distance(g, directed=T)

#the length all the shortest path
distances(g)


#clusetering
ceb <- cluster_edge_betweenness(g) 

dendPlot(ceb, mode="hclust")
plot(ceb, g) 

# based on propagating labels
clp <- cluster_label_prop(g)

plot(clp, g)

#greedy optimization of modularity
cfg <- cluster_fast_greedy(as.undirected(g))

plot(cfg, as.undirected(g))


#K-core decomposition
kc <- coreness(g, mode="all")

plot(g, vertex.size=kc*6, vertex.label=kc, vertex.color=colrs[kc])

####

#code inspired by:
#https://kateto.net/networks-r-igraph


