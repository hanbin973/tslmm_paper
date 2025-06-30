library(Matrix)
library(lme4breeding)

# load file
pedigreeGRM <- Matrix::readMM("spatial-simulation.pedigree.grm.mtx")
pedigreeGRM <- as.matrix(pedigreeGRM)
pheno <- scan("spatial-simulation.phenotypes.txt")
message("file loaded")
flush.console()

# process inputs
pheno <- pheno + rnorm(length(pheno))
id <- 1:length(pheno)
rownames(pedigreeGRM) <- colnames(pedigreeGRM) <- names(pheno) <- id
UD <- eigen(pedigreeGRM)
U <- UD$vectors
D <- diag(UD$values)
rownames(D) <- colnames(D) <- id
X <- model.matrix(~1, data=data.frame(pheno))
UX <- t(U) %*% X
UY <- t(U) %*% as.matrix(data.frame(pheno))
DTd <- data.frame(id, UY = UY, UX = UX[,1])
DTd$id <- as.character(DTd$id)
message("input processed")
flush.console()

# run mixed model
model <- lme4breeding::lmebreed(pheno ~ UX + (1|id),
                  relmat=list(id=D),
                  verbose = FALSE,
                  data=DTd)
message("model fitting done")
flush.console()

# process outputs
re <- lme4breeding::ranef(model)$id
blup <- U %*% re[as.character(id),]
path <- snakemake@output[["out_name"]]
write(blup, path, ncolumns=nrow(blup))
