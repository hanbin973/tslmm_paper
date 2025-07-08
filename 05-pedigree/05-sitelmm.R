library(Matrix)
library(data.table)
library(lme4breeding)

# load file
siteGRM <- data.table::fread(snakemake@input[["grm"]])
siteGRM <- as.matrix(siteGRM)
pheno_path <- snakemake@input[["pheno"]]
pheno <- rev(scan(pheno_path))
message("file loaded")
flush.console()

# process inputs
pheno <- pheno + rnorm(length(pheno))
id <- 1:length(pheno)
rownames(siteGRM) <- colnames(siteGRM) <- names(pheno) <- id
UD <- eigen(siteGRM)
U <- UD$vectors
D <- diag(pmax(UD$values, 0.00000001))
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
path <- snakemake@output[["blup"]]
write(blup, path, ncolumns=nrow(blup))
