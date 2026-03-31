struct _MockCB end  # to make BatchViewState without an NLPModel
MadNLP.create_array(::_MockCB, ::Type{T}, n::Int) where T = Vector{T}(undef, n)
_bvs(bs) = MadIPM.BatchViewState(_MockCB(), bs)

@testset "Batch views" begin

@testset "select_local!" begin
    bvs = _bvs(5)
    @test MadIPM.is_identity_view(MadIPM.root_view(bvs))

    MadIPM.select_local!(bvs, Int32[2, 4])
    @test MadIPM.active_view(bvs).local_to_root[1:2] == Int32[2, 4]

    MadIPM.select_local!(bvs, Int32[2])
    @test MadIPM.active_view(bvs).local_to_root[1] == Int32(4)
end

@testset "select_local! by bool" begin
    bvs = _bvs(5)
    MadIPM.select_local!(bvs, Bool[false, true, false, true, true])
    @test MadIPM.active_view(bvs).local_to_root[1:3] == Int32[2, 4, 5]
end

@testset "select_local! empty" begin
    bvs = _bvs(3)
    MadIPM.select_local!(bvs, Int32[])
    @test MadIPM.local_batch_size(MadIPM.active_view(bvs)) == 0
end

@testset "exclude_local!" begin
    bvs = _bvs(5)
    MadIPM.exclude_local!(bvs, Bool[false, true, false, true, false])
    @test MadIPM.active_view(bvs).local_to_root[1:3] == Int32[1, 3, 5]

    bvs = _bvs(3)
    MadIPM.exclude_local!(bvs, Bool[true, true, true])
    @test MadIPM.local_batch_size(MadIPM.active_view(bvs)) == 0

    bvs = _bvs(3)
    MadIPM.exclude_local!(bvs, Bool[false, false, false])
    @test MadIPM.is_identity_view(MadIPM.active_view(bvs))
end

@testset "restore_state! and reset_active_view!" begin
    bvs = _bvs(4)
    saved_root = MadIPM.select_local!(bvs, Int32[1, 3])
    saved_mid = MadIPM.select_local!(bvs, Int32[2])
    @test MadIPM.active_view(bvs).local_to_root[1] == Int32(3)

    MadIPM.restore_state!(bvs, saved_mid)
    @test MadIPM.active_view(bvs).local_to_root[1:2] == Int32[1, 3]

    MadIPM.restore_state!(bvs, saved_root)
    @test MadIPM.is_identity_view(MadIPM.active_view(bvs))

    MadIPM.select_local!(bvs, Int32[2, 3])
    MadIPM.reset_active_view!(bvs)
    @test MadIPM.is_identity_view(MadIPM.active_view(bvs))
end

@testset "fill_batch_view_mask!" begin
    bvs = _bvs(5)
    MadIPM.select_local!(bvs, Int32[2, 4])
    mask = zeros(Float64, 1, 5)
    MadIPM.fill_batch_view_mask!(mask, MadIPM.active_view(bvs))
    @test mask == [0.0 1.0 0.0 1.0 0.0]
end

@testset "local_to_root_dev" begin
    bvs = _bvs(4)
    MadIPM.select_local!(bvs, Int32[2, 4])
    @test MadIPM.local_to_root_dev(MadIPM.active_view(bvs))[1:2] == Int32[2, 4]

    MadIPM.select_local!(bvs, Int32[1])
    @test MadIPM.local_to_root_dev(MadIPM.active_view(bvs))[1] == Int32(2)
end

@testset "gather/scatter/compact" begin
    bvs = _bvs(5)
    MadIPM.select_local!(bvs, Int32[2, 4])
    v = MadIPM.active_view(bvs)
    src = reshape(collect(1.0:15.0), 3, 5)

    gathered = zeros(3, 2)
    MadIPM.gather_batch_view_columns!(gathered, src, v)
    @test gathered == src[:, [2, 4]]

    dst = fill(-1.0, 3, 5)
    MadIPM.scatter_batch_view_columns!(dst, gathered, v)
    @test dst[:, [2, 4]] == src[:, [2, 4]]
    @test all(dst[:, [1, 3, 5]] .== -1.0)

    data = copy(src)
    MadIPM.compact_active_columns_inplace!(data, v)
    @test data[:, 1:2] == src[:, [2, 4]]
end

@testset "gather then scatter round-trips" begin
    bvs = _bvs(5)
    MadIPM.select_local!(bvs, Int32[1, 3, 5])
    v = MadIPM.active_view(bvs)

    src = reshape(collect(1.0:10.0), 2, 5)
    gathered = zeros(2, 3)
    MadIPM.gather_batch_view_columns!(gathered, src, v)
    dst = zeros(2, 5)
    MadIPM.scatter_batch_view_columns!(dst, gathered, v)
    @test dst[:, [1, 3, 5]] == src[:, [1, 3, 5]]
    @test dst[:, [2, 4]] == zeros(2, 2)
end

end
