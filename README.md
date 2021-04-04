# Retirement Eggs and Baskets 
Lifecycle model of portfolio allocation with mortgaging, housing and pension choice of
Bateman et al (2021)

The model solves the non-convex, non-smooth dynamic problem using sub-derivative
Euler equations of the convexified bi-conjugate of the problem (see Dobrescu and Shanker, 2021) and employing the endogenous grid method. 

Model parameters then estimated using cross-entropy method, with each parameter draw solved in 
parallel in groups of N = 2000 groups 6 cores.

