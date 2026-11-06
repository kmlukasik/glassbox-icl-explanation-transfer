def log_run(cfg: dict, aucs: dict, results, importance) -> None:
    """Zapisuje przebieg w W&B (jeśli włączone w configu)"""
    w = cfg.get("wandb", {})
    if not w.get("enabled", False):
        return
    import wandb

    run = wandb.init(
        project=w["project"],
        mode=w.get("mode", "online"),
        config=cfg,
        name=f"{cfg['dataset']}-seed{cfg['seed']}",
    )
    run.summary.update(aucs)  # {"ebm_auc": ..., "tabpfn_auc": ...}
    for row in results.to_dict("records"):
        role = row["role"]
        run.summary.update(
            {f"{role}/{k}": v for k, v in row.items() if isinstance(v, (int, float))}
        )
    run.log({
        "results": wandb.Table(dataframe=results),
        "ebm_importance": wandb.Table(
            dataframe=importance.rename_axis("feature").reset_index(name="importance")
        ),
    })
    run.finish()