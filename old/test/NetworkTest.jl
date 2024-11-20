using Test: @testset, @test

include("../src/Network.jl")
using .Network: create_weighted_cycle_graph, mat_nv, mat_ne, mat_degree, mat_update_weight!

@testset "create_weighted_cycle_graph" begin
    mat = create_weighted_cycle_graph(10, 4, 0.5)
    @test mat_nv(mat) == 10
    @test mat_ne(mat) == 20
    @test mat_degree(mat) == fill(4, 10)
    
    mat_update_weight!(mat, 1, 6, 0.1)
    @test mat_nv(mat) == 10
    @test mat_ne(mat) == 21
    @test mat_degree(mat) == [5, 4, 4, 4, 4, 5, 4, 4, 4, 4]

    mat_update_weight!(mat, 3, 4, 0.0)
    mat_update_weight!(mat, 8, 9, 0.0)
    @test mat_nv(mat) == 10
    @test mat_ne(mat) == 19
    @test mat_degree(mat) == [5, 4, 3, 3, 4, 5, 4, 3, 3, 4]
end
