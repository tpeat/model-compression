# new loss function for KL Divergence
import torch
import torch.nn.functional as F

class DistillationLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, temperature=4):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_outputs, teacher_outputs, labels):
        # Calculate KL Divergence for distillation loss
        soft_labels = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_labels, teacher_soft, reduction='batchmean') * (self.temperature ** 2)

        # Calculate standard cross-entropy loss
        classification_loss = F.cross_entropy(student_outputs, labels)

        # Combine losses
        loss = self.alpha * distillation_loss + (1 - self.alpha) * classification_loss
        return loss