# Sourced algorithm settings; historical values are preserved.
GA_ARL_METHOD_ARGS=(
    +project_kwargs.sample_remain_num=-1
    +project_kwargs.use_loss_for_p_star=False
    +project_kwargs.select_by=ref_no_var
    +replay.enable=True
    +replay.k=4
    +replay.pmis=True
    +replay.warmup=5
)
