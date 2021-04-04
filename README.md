# Retirement Eggs and Baskets 
Lifecycle model of portfolio allocation with mortgaging, renting, housing and pension choice
(Bateman et al, 2021)

Model features non-convex housing adjustment friction, discrete rental/ home-owner choice, non-ownder occupied housing, mortgaging constrained by collateral, discrete pension plan and pension risk choice and voluntary and compulsory pension contributions. 

Code solves the non-convex, non-smooth dynamic problem using sub-derivative
Euler equations of the convexified bi-conjugate of the problem (see Dobrescu and Shanker, 2021) and then employing the endogenous grid method. 

Model parameters then estimated using cross-entropy method, with each parameter draw solved in 
parallel in groups of N = 2000 groups 6 cores.

