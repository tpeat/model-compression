# Model Compression

Objective:
- freeze all weights from large model with cell/block structure
- swap out module for smaller or easier to parrellize module
- train for epoch
- evaluate keep original module or not
- iterate all layers