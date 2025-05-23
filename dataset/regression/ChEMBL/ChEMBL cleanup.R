#load required packages

  library(dplyr)
  library(ampir)
  library(stringr)
  library(tidyr)
  
#Three documents were downloaded from ChEMBL website
  #1. All small molecules - "ChEMBL_all small molecules.csv"
  #2. All Kd values - "ChEMBL kd all.csv"
  #3. All ChEMBL fasta data - "chembl_28.fa"
  
#1. Keep only the Kd data of ("small molecules" & Molecular weight<900)

  #1.1 Rechieve all the ChEMBL ID of small molecules
      chem_sm_id<-read.csv("ChEMBL_all small molecules.csv",
                         header = TRUE,sep=";",
                         colClasses = c(NA, rep("NULL",30)))
  
  #1.2 read in Kd data and cleanup
      chem<-read.csv("ChEMBL kd all.csv",sep=";")
      class(chem$Molecular.Weight)<-"numeric"
  
  #1.3 keep Kd data for molecules whose MW <900

      chem<-chem[chem$Molecular.Weight<900,]
  
  #1.4 Keep Kd data for small molecules
      chem_sm<-chem[chem$Molecule.ChEMBL.ID %in% 
                      chem_sm_id$ChEMBL.ID,]
      chem_sm<-chem_sm[!is.na(chem_sm$Standard.Value),]
      chem_sm<-chem_sm[chem_sm$Standard.Units=="nM",]
      chem_sm6<-chem_sm[,c(1,8,11,13,35)]
  
  #1.5 Add target sequence to the table
      chem_seq<-read_faa("chembl_28.fa")
      chem_seq$Target.ChEMBL.ID<-str_split_fixed(chem_seq$seq_name," ",3)[,2]
      chem_seq$seq_name<-NULL
      chem_seq_u<-separate_rows(chem_seq,Target.ChEMBL.ID,sep = ",")
      chem_seq_u<-chem_seq_u%>% distinct(Target.ChEMBL.ID,.keep_all = TRUE)
      chem_all<-merge(chem_sm6,chem_seq_u,all.x = TRUE)
  
  #1.6 order the table according to sequence
      chem_sort<-chem_all[(order(chem_all$seq_aa)),]
  
#2. Prepare datafile for target sequence
  chem_target<-chem_sort[,c(1,6)]
  chem_target_u<-chem_target %>% 
      distinct(seq_aa,.keep_all = TRUE)
  chem_target_u<-chem_target_u[!is.na(chem_target_u$seq_aa),]
  write.table(chem_target_u,"ChEMBL_Target_Sequence.txt")
  
#3. Prepare a huge list for all the data, group by target sequence  
  chem_list<-split.data.frame(chem_sort,chem_sort$seq_aa)
  chem_temp<-matrix(nrow=nrow(chem_target_u),ncol = 6)
  
      for (i in 1:length(chem_list)){
        seq<-chem_target_u[i,2]
        list<-chem_list[seq]
        for (j in 1:6){
          chem_temp[i,j]<-paste(unlist(lapply(list,"[[",j)),
                                sep = " ",collapse = " ")
        }
      }
  chem_all_u<-chem_target_u
  chem_all_u$Ligand.SMILES<-chem_temp[,3]
  chem_all_u$Ligand.CHEMBLID<-chem_temp[,2]
  chem_all_u$Kd.nM<-chem_temp[,4]
  chem_all_u$Affinity<-chem_temp[,5]
  
#4 Prepare datafile for all SMILES
    chemsmiles<-chem_all_u[,c(3,4)]
    write.table(chemsmiles,"Chem_SMILES.txt",row.names = FALSE)

#5 Prepare datafile for all Kd & Affinity
    chemkd<-chem_all_u[,5]
    write.csv(chemkd,"Chem_Kd_nM.txt",row.names = FALSE)
    
    chemaffnity<-chem_all_u[,6]
    write.csv(chemaffnity,"Chem_Affinity.txt",row.names = FALSE)

