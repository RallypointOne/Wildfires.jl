@testset "hamada_rates" begin
    h = HamadaModel()
    down, side, up = hamada_rates(h, 15.0, 15.0, 10.0, 0.0)
    @test down > side > up > 0
    @test down ≈ 0.274 rtol = 1e-2        # (A + D) / T₄ by hand: 25 m / 1.52 min
    @test all(isfinite, (down, side, up))

    # Calm: the HAZUS blend makes spread isotropic.
    calm = hamada_rates(h, 0.0, 15.0, 10.0, 0.0)
    @test calm[1] ≈ calm[2] ≈ calm[3]
    @test calm[1] > 0

    # Fire-resistant buildings slow the fire; wind speeds it up. Wider
    # separation raises the rate, since the front advances one block pitch
    # per building burn time.
    @test hamada_rates(h, 15.0, 15.0, 10.0, 1.0)[1] < down
    @test hamada_rates(h, 25.0, 15.0, 10.0, 0.0)[1] > down
    @test hamada_rates(h, 15.0, 15.0, 40.0, 0.0)[1] > down

    # Degenerate inputs stay finite and positive.
    @test all(x -> isfinite(x) && x > 0, hamada_rates(h, 0.0, 0.0, 0.0, 0.0))
    @test all(x -> isfinite(x) && x > 0, hamada_rates(h, -5.0, 15.0, 10.0, 2.0))

    h32 = HamadaModel(Float32)
    @test hamada_rates(h32, 15f0, 15f0, 10f0, 0f0) isa NTuple{3, Float32}
    @test hamada_rates(HamadaModel{Float32}(h), 15.0, 15.0, 10.0, 0.0)[1] ≈ down rtol = 1e-5
    @test HamadaModel(; c5 = (2.0, 0.25, 0.2)).c5[1] == 2.0
end

@testset "ellipse_speed" begin
    head, flank, back = 2.0, 0.5, 0.2
    @test ellipse_speed(1.0, 0.0, 1.0, 0.0, head, flank, back) ≈ head
    @test ellipse_speed(-1.0, 0.0, 1.0, 0.0, head, flank, back) ≈ back
    @test ellipse_speed(0.0, 1.0, 1.0, 0.0, head, flank, back) ≈ flank
    @test ellipse_speed(0.0, -1.0, 1.0, 0.0, head, flank, back) ≈ flank
    # Rotating the wind rotates the shape.
    @test ellipse_speed(0.0, 1.0, 0.0, 1.0, head, flank, back) ≈ head
    # Between the head and the flank, and never below the flank rate.
    s45 = ellipse_speed(√0.5, √0.5, 1.0, 0.0, head, flank, back)
    @test flank < s45 < head
    # Calm wind is a zero vector: the flank rate everywhere.
    @test ellipse_speed(1.0, 0.0, 0.0, 0.0, head, flank, back) ≈ flank
end
