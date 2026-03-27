@testset "Iteration-level state matching" begin

    # ──────────────────────────────────────────────────────────
    # init_starting_point! explicit snapshot: verify post-init state matches
    # ──────────────────────────────────────────────────────────
    @testset "init snapshot: $label" for (label, make_qp) in ALL_TEST_PROBLEMS
        qp = make_qp()
        seq = build_seq(qp)
        bat = build_batch(qp)
        assert_init_match(seq, bat, 1)
    end

    # ──────────────────────────────────────────────────────────
    # batch_size=1: every problem, full iteration trace
    # ──────────────────────────────────────────────────────────
    @testset "batch_size=1: $label" for (label, make_qp) in ALL_TEST_PROBLEMS
        qp = make_qp()
        seq = build_seq(qp)
        bat = build_batch(qp)
        run_iterations_bs1!(seq, bat, 50)
    end

    # ──────────────────────────────────────────────────────────
    # batch_size=2: two DIFFERENT problems in one batch
    # ──────────────────────────────────────────────────────────
    @testset "batch_size=2: $label" for (label, make_paired) in PAIRED_PROBLEMS
        qp_a, qp_b = make_paired()
        seq1 = build_seq(qp_a)
        seq2 = build_seq(qp_b)
        bat = build_batch_from_qps([qp_a, qp_b])
        run_iterations_bs2!(seq1, seq2, bat, 50)
        @test seq1.status == MadNLP.SOLVE_SUCCEEDED
        @test seq2.status == MadNLP.SOLVE_SUCCEEDED
    end

    # ──────────────────────────────────────────────────────────
    # Scaling verification: confirm non-unit scales are active
    # ──────────────────────────────────────────────────────────
    @testset "scaling verification" begin
        @testset "$label" for (label, make_qp) in [
            ("scaled_qp", _setup_scaled_qp),
            ("scaled_ineq", _setup_scaled_ineq_qp),
        ]
            qp = make_qp()
            seq = build_seq(qp)
            bat = build_batch(qp)
            # Verify obj_scale < 1.0
            @test seq.cb.obj_scale[] < 1.0
            @test bat.bcb.obj_scale[1] < 1.0
            # Verify con_scale has non-unit entries
            @test minimum(seq.cb.con_scale) < 1.0
            @test minimum(bat.bcb.con_scale[:, 1]) < 1.0
            # Verify scales match between sequential and batch
            @test abs(seq.cb.obj_scale[] - bat.bcb.obj_scale[1]) < 1e-12
            @test cmp(seq.cb.con_scale, bat.bcb.con_scale[:, 1]) < 1e-12
        end
    end

    # ──────────────────────────────────────────────────────────
    # Fixed variable verification
    # ──────────────────────────────────────────────────────────
    @testset "fixed variable verification" begin
        qp = _setup_fixed_var_qp()
        seq = build_seq(qp)
        bat = build_batch(qp)
        # Verify MakeParameter handler is active for sequential
        @test seq.cb.fixed_handler isa MadNLP.MakeParameter
        @test bat.bcb.fixed_handler isa MadNLP.MakeParameter
        # Sequential removes the fixed var from nvar; batch may keep it
        @test MadNLP.n_variables(seq.cb) < 3  # reduced from 3 vars
        # Run iterations (also part of ALL_TEST_PROBLEMS bs1 above, but
        # this standalone test documents the fixed-variable intent)
        run_iterations_bs1!(seq, bat, 50)
    end

    # ──────────────────────────────────────────────────────────
    # Staggered convergence: one instance converges much earlier
    # with explicit zero_inactive_step! and active mask verification
    # ──────────────────────────────────────────────────────────
    @testset "staggered convergence (bs=2)" begin
        qp_easy, qp_hard = _paired_staggered()
        seq1 = build_seq(qp_easy)
        seq2 = build_seq(qp_hard)
        bat = build_batch_from_qps([qp_easy, qp_hard])

        # Run iterations manually to verify zero_inactive_step! and active mask
        ws = bat.workspace
        seq1_done = false
        seq2_done = false
        inactive_verified = false

        for iter in 1:50
            if !seq1_done; MadIPM.update_termination_criteria!(seq1); end
            if !seq2_done; MadIPM.update_termination_criteria!(seq2); end
            MadIPM.update_termination_criteria!(bat)

            prev_seq1_done = seq1_done
            prev_seq2_done = seq2_done
            seq1_done = seq1_done || MadIPM.is_done(seq1)
            seq2_done = seq2_done || MadIPM.is_done(seq2)

            MadIPM.update_active_set!(bat)
            active = MadIPM.active_view(bat.batch_views)
            MadIPM.local_batch_size(active) == 0 && break
            MadIPM._update_active_mask!(bat)

            # Verify active mask matches expected state
            if seq1_done && !seq2_done
                @test Int[active.local_to_root[i] for i in 1:active.n] == [2]
                @test ws.active_mask[1] == 0.0
                @test ws.active_mask[2] == 1.0
            elseif !seq1_done && seq2_done
                @test Int[active.local_to_root[i] for i in 1:active.n] == [1]
                @test ws.active_mask[1] == 1.0
                @test ws.active_mask[2] == 0.0
            elseif !seq1_done && !seq2_done
                @test Int[active.local_to_root[i] for i in 1:active.n] == [1, 2]
            end

            (seq1_done && seq2_done) && break

            # Factorize
            if !seq1_done; MadIPM.factorize_system!(seq1); end
            if !seq2_done; MadIPM.factorize_system!(seq2); end
            MadIPM.factorize_system!(bat)

            # Prediction + Mehrotra
            if !seq1_done; MadIPM.prediction_step!(seq1); end
            if !seq2_done; MadIPM.prediction_step!(seq2); end
            MadIPM.prediction_step!(bat)
            if !seq1_done; MadIPM.mehrotra_correction_direction!(seq1); end
            if !seq2_done; MadIPM.mehrotra_correction_direction!(seq2); end
            MadIPM.mehrotra_correction_direction!(bat)

            # Update step
            if !seq1_done; MadIPM.update_step!(seq1.opt.step_rule, seq1); end
            if !seq2_done; MadIPM.update_step!(seq2.opt.step_rule, seq2); end
            MadIPM.update_step!(bat.opt.step_rule, bat)
            MadIPM.zero_inactive_step!(bat)

            # Verify zero_inactive_step! zeroed the inactive instance
            if seq1_done && !seq2_done
                @test ws.alpha_p[1] == 0.0
                @test ws.alpha_d[1] == 0.0
                @test ws.alpha_p[2] > 0.0  # active instance has nonzero step
                inactive_verified = true
            elseif !seq1_done && seq2_done
                @test ws.alpha_p[2] == 0.0
                @test ws.alpha_d[2] == 0.0
                @test ws.alpha_p[1] > 0.0
                inactive_verified = true
            end
            if !seq1_done; assert_step_match(seq1, bat, 1); end
            if !seq2_done; assert_step_match(seq2, bat, 2); end

            # Apply step + evaluate model
            if !seq1_done; MadIPM.apply_step!(seq1); end
            if !seq2_done; MadIPM.apply_step!(seq2); end
            MadIPM.apply_step!(bat)
            if !seq1_done; MadIPM.evaluate_model!(seq1); end
            if !seq2_done; MadIPM.evaluate_model!(seq2); end
            MadIPM.evaluate_model!(bat)
        end

        @test seq1.status == MadNLP.SOLVE_SUCCEEDED
        @test seq2.status == MadNLP.SOLVE_SUCCEEDED
        # Verify the easy one converged earlier
        @test seq1.cnt.k < seq2.cnt.k
        # Verify we actually tested the inactive masking path
        @test inactive_verified
    end

    # ──────────────────────────────────────────────────────────
    # batch_size=4: four DIFFERENT problems in one batch
    # ──────────────────────────────────────────────────────────
    @testset "batch_size=4: $label" for (label, make_quad) in [
        ("lower-only LP", _quad_lower_only),
        ("doubly-bounded QP", _quad_doubly_bounded),
        ("dense Hessian QP", _quad_dense_hess),
    ]
        qp1, qp2, qp3, qp4 = make_quad()
        seqs = [build_seq(qp) for qp in [qp1, qp2, qp3, qp4]]
        bat = build_batch_from_qps([qp1, qp2, qp3, qp4])
        run_iterations_bsN!(seqs, bat, 50)
        for s in seqs
            @test s.status == MadNLP.SOLVE_SUCCEEDED
        end
    end

    # ──────────────────────────────────────────────────────────
    # Factorization retry: verify retry loop works for both seq and batch
    # ──────────────────────────────────────────────────────────
    @testset "factorization retry path" begin
        qp = _setup_doubly_bounded_qp()
        seq = build_seq(qp)
        bat = build_batch(qp)

        # Artificially set del_w/del_c to large values (as if retry triggered)
        seq.del_w = 1e-4
        seq.del_c = -1e-4
        bat.del_w .= 1e-4
        bat.del_c .= -1e-4

        MadIPM.factorize_system!(seq)
        MadIPM.factorize_system!(bat)
        assert_regularization_match(seq, bat, 1)
        assert_kkt_diagonals_match(seq, bat, 1)
        assert_kkt_matrix_match(seq, bat, 1)

        MadIPM.prediction_step!(seq)
        MadIPM.prediction_step!(bat)
        assert_prediction_match(seq, bat, 1)
    end

    # ──────────────────────────────────────────────────────────
    # KKT mul! verification: batch scatter mul vs sequential sparse mul
    # ──────────────────────────────────────────────────────────
    @testset "KKT mul! $label" for (label, make_qp) in [
        ("QP (nlb>0, nub>0)", _setup_small_qp),
        ("QP doubly-bounded", _setup_doubly_bounded_qp),
        ("QP dense Hessian+mixed", _setup_dense_hess_mixed_qp),
        ("QP mixed bounds", _setup_mixed_bounds_qp),
    ]
        qp = make_qp()
        seq = build_seq(qp)
        bat = build_batch(qp)

        # Set up KKT diagonals (needed for mul!)
        MadIPM.update_regularization!(seq, seq.opt.regularization)
        MadIPM.set_aug_diagonal_reg!(seq.kkt, seq)
        MadIPM.update_regularization!(bat, bat.opt.regularization)
        MadIPM.set_aug_diagonal_reg!(bat.kkt, bat)

        # Use the current d as input vector (has non-trivial values after init)
        # Copy d → p to have a known input
        copyto!(MadNLP.full(seq.p), MadNLP.full(seq.d))
        copyto!(MadNLP.full(bat.p), MadNLP.full(bat.d))

        # Compute mul!(w1, kkt, p) for both
        seq_w = seq._w1
        bat_w = bat._w1
        fill!(MadNLP.full(seq_w), 0.0)
        fill!(MadNLP.full(bat_w), 0.0)
        mul!(seq_w, seq.kkt, seq.p)
        mul!(bat_w, bat.kkt, bat.p)

        @test cmp(MadNLP.full(seq_w), coln(MadNLP.full(bat_w), 1)) < 1e-10
    end

    # ──────────────────────────────────────────────────────────
    # Step rule variants (bs=1, with _setup_small_qp)
    # ──────────────────────────────────────────────────────────
    @testset "step_rule=$label" for (label, step_rule) in [
        ("ConservativeStep", MadIPM.ConservativeStep(0.995)),
        ("AdaptiveStep", MadIPM.AdaptiveStep(0.99)),
        ("MehrotraAdaptiveStep", MadIPM.MehrotraAdaptiveStep(0.99)),
    ]
        qp = _setup_small_qp()
        seq = build_seq(qp; step_rule=step_rule)
        bat = build_batch(qp; step_rule=step_rule)
        run_iterations_bs1!(seq, bat, 50)
    end

    # ──────────────────────────────────────────────────────────
    # Step rule variants with diverse problem types
    # ──────────────────────────────────────────────────────────
    @testset "step_rule=$sr_label on $prob_label" for (sr_label, step_rule) in [
        ("ConservativeStep", MadIPM.ConservativeStep(0.995)),
        ("AdaptiveStep", MadIPM.AdaptiveStep(0.99)),
        ("MehrotraAdaptiveStep", MadIPM.MehrotraAdaptiveStep(0.99)),
    ], (prob_label, make_qp) in [
        ("free vars", _setup_free_qp),
        ("all-ineq", _setup_all_ineq_qp),
        ("doubly-bounded", _setup_doubly_bounded_qp),
        ("dense Hessian+mixed", _setup_dense_hess_mixed_qp),
    ]
        qp = make_qp()
        seq = build_seq(qp; step_rule=step_rule)
        bat = build_batch(qp; step_rule=step_rule)
        run_iterations_bs1!(seq, bat, 50)
    end

    # ──────────────────────────────────────────────────────────
    # Regularization variants (bs=1, with _setup_small_qp)
    # ──────────────────────────────────────────────────────────
    @testset "regularization=$label" for (label, make_reg) in [
        ("NoRegularization", () -> MadIPM.NoRegularization()),
        ("FixedRegularization", () -> MadIPM.FixedRegularization(1e-10, 1e-10)),
        ("AdaptiveRegularization", () -> MadIPM.AdaptiveRegularization(1e-6, 1e-6, 1e-12)),
    ]
        qp = _setup_small_qp()
        seq = build_seq(qp; regularization=make_reg())
        bat = build_batch(qp; regularization=make_reg())
        run_iterations_bs1!(seq, bat, 50)
    end

    # ──────────────────────────────────────────────────────────
    # Regularization variants with diverse problem types
    # ──────────────────────────────────────────────────────────
    @testset "regularization=$reg_label on $prob_label" for (reg_label, make_reg) in [
        ("NoRegularization", () -> MadIPM.NoRegularization()),
        ("AdaptiveRegularization", () -> MadIPM.AdaptiveRegularization(1e-6, 1e-6, 1e-12)),
    ], (prob_label, make_qp) in [
        ("free vars", _setup_free_qp),
        ("all-ineq", _setup_all_ineq_qp),
        ("doubly-bounded", _setup_doubly_bounded_qp),
        ("dense Hessian+mixed", _setup_dense_hess_mixed_qp),
    ]
        qp = make_qp()
        seq = build_seq(qp; regularization=make_reg())
        bat = build_batch(qp; regularization=make_reg())
        run_iterations_bs1!(seq, bat, 50)
    end

    # ──────────────────────────────────────────────────────────
    # Step rule + regularization variants with bs=2
    # ──────────────────────────────────────────────────────────
    @testset "options bs=2: $label" for (label, make_opts) in [
        ("Conservative+NoReg", () -> (step_rule=MadIPM.ConservativeStep(0.995), regularization=MadIPM.NoRegularization())),
        ("Mehrotra+AdaptiveReg", () -> (step_rule=MadIPM.MehrotraAdaptiveStep(0.99), regularization=MadIPM.AdaptiveRegularization(1e-6, 1e-6, 1e-12))),
    ]
        qp_a, qp_b = _paired_doubly_bounded()
        # Each solver needs its own copy of mutable options (e.g. AdaptiveRegularization)
        opts1 = make_opts()
        opts2 = make_opts()
        opts_bat = make_opts()
        seq1 = build_seq(qp_a; opts1...)
        seq2 = build_seq(qp_b; opts2...)
        bat = build_batch_from_qps([qp_a, qp_b]; opts_bat...)
        run_iterations_bs2!(seq1, seq2, bat, 50)
        @test seq1.status == MadNLP.SOLVE_SUCCEEDED
        @test seq2.status == MadNLP.SOLVE_SUCCEEDED
    end

    # ──────────────────────────────────────────────────────────
    # Combined step+reg options with diverse bs=2 paired problems
    # ──────────────────────────────────────────────────────────
    @testset "options bs=2: $opt_label on $prob_label" for (opt_label, make_opts) in [
        ("Conservative+NoReg", () -> (step_rule=MadIPM.ConservativeStep(0.995), regularization=MadIPM.NoRegularization())),
        ("Adaptive+AdaptiveReg", () -> (step_rule=MadIPM.AdaptiveStep(0.99), regularization=MadIPM.AdaptiveRegularization(1e-6, 1e-6, 1e-12))),
    ], (prob_label, make_paired) in [
        ("dense-hess", _paired_dense_hess),
        ("all-ineq", _paired_all_ineq),
        ("mixed-bounds", _paired_mixed_bounds),
    ]
        qp_a, qp_b = make_paired()
        opts1 = make_opts()
        opts2 = make_opts()
        opts_bat = make_opts()
        seq1 = build_seq(qp_a; opts1...)
        seq2 = build_seq(qp_b; opts2...)
        bat = build_batch_from_qps([qp_a, qp_b]; opts_bat...)
        run_iterations_bs2!(seq1, seq2, bat, 50)
        @test seq1.status == MadNLP.SOLVE_SUCCEEDED
        @test seq2.status == MadNLP.SOLVE_SUCCEEDED
    end

    # ──────────────────────────────────────────────────────────
    # init_regularization! explicit comparison
    # ──────────────────────────────────────────────────────────
    @testset "init_regularization! $label" for (label, make_reg) in [
        ("NoRegularization", () -> MadIPM.NoRegularization()),
        ("FixedRegularization", () -> MadIPM.FixedRegularization(1e-10, 1e-10)),
        ("AdaptiveRegularization", () -> MadIPM.AdaptiveRegularization(1e-6, 1e-6, 1e-12)),
    ]
        qp = _setup_small_qp()
        seq = build_seq(qp; regularization=make_reg())
        bat = build_batch(qp; regularization=make_reg())
        # After build, init_regularization! has been called.
        # del_w and del_c should match.
        assert_regularization_match(seq, bat, 1)
    end

    # ══════════════════════════════════════════════════════════
    # BatchQuadraticModel tests (different H/A values per instance)
    # ══════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────
    # BatchQuadraticModel bs=1: verify same results as ObjRHSBatch
    # ──────────────────────────────────────────────────────────
    @testset "fullbatch bs=1: $label" for (label, make_qp) in ALL_TEST_PROBLEMS
        qp = make_qp()
        seq = build_seq(qp)
        bat = build_fullbatch(qp)
        assert_init_match(seq, bat, 1)
        run_iterations_bs1!(seq, bat, 50)
    end

    # ──────────────────────────────────────────────────────────
    # BatchQuadraticModel bs=2: different H AND A values
    # ──────────────────────────────────────────────────────────
    @testset "fullbatch bs=2: $label" for (label, make_paired) in FULLBATCH_PAIRED_PROBLEMS
        qp_a, qp_b = make_paired()
        seq1 = build_seq(qp_a)
        seq2 = build_seq(qp_b)
        bat = build_fullbatch_from_qps([qp_a, qp_b])
        run_iterations_bs2!(seq1, seq2, bat, 50)
        @test seq1.status == MadNLP.SOLVE_SUCCEEDED
        @test seq2.status == MadNLP.SOLVE_SUCCEEDED
    end

    # ──────────────────────────────────────────────────────────
    # BatchQuadraticModel scaling: verify per-instance scaling with different A
    # ──────────────────────────────────────────────────────────
    @testset "fullbatch scaling verification" begin
        qp_a, qp_b = _fullbatch_paired_scaled()
        seq1 = build_seq(qp_a)
        seq2 = build_seq(qp_b)
        bat = build_fullbatch_from_qps([qp_a, qp_b])
        # Verify non-unit scales
        @test seq1.cb.obj_scale[] < 1.0
        @test seq2.cb.obj_scale[] < 1.0
        @test bat.bcb.obj_scale[1] < 1.0
        @test bat.bcb.obj_scale[2] < 1.0
        # Verify per-instance scales match sequential
        @test abs(seq1.cb.obj_scale[] - bat.bcb.obj_scale[1]) < 1e-12
        @test abs(seq2.cb.obj_scale[] - bat.bcb.obj_scale[2]) < 1e-12
        @test cmp(seq1.cb.con_scale, bat.bcb.con_scale[:, 1]) < 1e-12
        @test cmp(seq2.cb.con_scale, bat.bcb.con_scale[:, 2]) < 1e-12
        # Verify that the two instances have DIFFERENT scales (since A values differ)
        @test bat.bcb.con_scale[1, 1] != bat.bcb.con_scale[1, 2]
    end

    # ──────────────────────────────────────────────────────────
    # BatchQuadraticModel with options variants
    # ──────────────────────────────────────────────────────────
    @testset "fullbatch options: $label" for (label, make_opts) in [
        ("Conservative+NoReg", () -> (step_rule=MadIPM.ConservativeStep(0.995), regularization=MadIPM.NoRegularization())),
        ("Adaptive+AdaptiveReg", () -> (step_rule=MadIPM.AdaptiveStep(0.99), regularization=MadIPM.AdaptiveRegularization(1e-6, 1e-6, 1e-12))),
    ]
        qp_a, qp_b = _fullbatch_paired_dense_hess()
        opts1 = make_opts()
        opts2 = make_opts()
        opts_bat = make_opts()
        seq1 = build_seq(qp_a; opts1...)
        seq2 = build_seq(qp_b; opts2...)
        bat = build_fullbatch_from_qps([qp_a, qp_b]; opts_bat...)
        run_iterations_bs2!(seq1, seq2, bat, 50)
        @test seq1.status == MadNLP.SOLVE_SUCCEEDED
        @test seq2.status == MadNLP.SOLVE_SUCCEEDED
    end
end
