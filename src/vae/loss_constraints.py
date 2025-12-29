# loss_constraints.py
import torch

def constraint_loss(T_pred, T_min, T_max, margin=0.05, mode="quadratic"):
    """
    Penaliza predicciones de temperatura fuera del rango físico [T_min, T_max].

    Args:
        T_pred (Tensor): Temperaturas normalizadas predichas (B, 1) o (B,)
        T_min (float): Límite inferior del rango válido
        T_max (float): Límite superior del rango válido
        margin (float): Tolerancia adicional antes de penalizar
        mode (str): 'quadratic' (por defecto) o 'linear' para control de suavidad

    Returns:
        Tensor: Escalar con la pérdida de restricción (promedio sobre el batch)
    """
    if T_pred.dim() > 1:
        T_pred = T_pred.squeeze(-1)

    # Penalización por debajo del mínimo
    under = torch.relu((T_min - margin) - T_pred)
    # Penalización por encima del máximo
    over = torch.relu(T_pred - (T_max + margin))

    if mode == "quadratic":
        loss_under = under.pow(2)
        loss_over = over.pow(2)
    elif mode == "linear":
        loss_under = under
        loss_over = over
    else:
        raise ValueError("mode must be 'quadratic' or 'linear'")

    loss = (loss_under + loss_over).mean()
    return loss


def physical_loss_summary(T_pred, T_min, T_max):
    """
    Diagnóstico útil: devuelve qué % del batch está fuera de los límites.
    No se usa en entrenamiento, pero sirve para debug o métricas físicas.
    """
    with torch.no_grad():
        below = (T_pred < T_min).float().mean().item()
        above = (T_pred > T_max).float().mean().item()
    return {"below_min": below, "above_max": above}
