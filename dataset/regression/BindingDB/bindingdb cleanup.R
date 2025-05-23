#Load required packages
    setwd("~/Desktop/R")
    library(dplyr)

#All bindingdb data was downloaded from 
    #https://www.bindingdb.org/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_2021m2.tsv.zip  

#1. Cleanup the bindingdb data
    #rread in bindingdb data(The first 39 columns)
    bdb<-read.csv("BindingDB_All.tsv",header = TRUE, 
                  sep = "\t", 
                  colClasses = c(rep(NA,39),rep("NULL",154)))
   
    #Select the desired columns only
    bdb5<-bdb[,c(2,11,36,38,39)]
    class(bdb5$Kd..nM.)<-"numeric"
   
    #Remove all the rows without Kd data
    bdb5<-bdb5[!is.na(bdb5$Kd..nM.),]

    #Force all Kd values>10000 to 10000
    bdb5[bdb5$Kd..nM.>10000,][2]<-10000

    #Sort and clean up the columns
    bdb5$Kd.nM<-bdb5$Kd..nM.
    bdb5$Kd..nM.<-NULL
    colnames(bdb5)<-
      c("Ligands.SMILES","Ligands.ZINCID",
        "Target.Sequence","Target.PDBID","Kd.nM")
    bdb5_sort<-bdb5[(order(bdb5$Target.Sequence)),]
    
#2. Prepare datafile for target sequence
    bdb5target<-bdb5_sort[,c(3,4)]

    bdb5target_u<-bdb5target %>% 
      distinct(Target.Sequence,.keep_all = TRUE)
    
    write.table(bdb5target_u,"BindingDB_Target_Sequence.txt")

#3. Prepare a big list grouped by target sequence
    bdb5_list<-split.data.frame(bdb5_sort,bdb5_sort$Target.Sequence)
    bdb5_temp<-matrix(nrow=nrow(bdb5target_u),ncol = 5)
    
    for (i in 1:length(bdb5_list)){
    seq<-bdb5target_u[i,1]
    list<-bdb5_list[seq]
         for (j in 1:5){
           bdb5_temp[i,j]<-paste(unlist(lapply(list,"[[",j)),
                                 sep = " ",collapse = " ")
         }
    }
    
    bdb5_all_u<-bdb5target_u
    bdb5_all_u$Ligands.SMILES<-bdb5_temp[,1]
    bdb5_all_u$Ligands.ZINCID<-bdb5_temp[,2]
    bdb5_all_u$Kd.nM<-bdb5_temp[,5]
    
#4. Prepare datafile for all SMILES
    bdb3smiles<-bdb5_all_u[,c(3,4)]
    write.table(bdb3smiles,"BindingDB_SMILES.txt",row.names = FALSE)
    
#5. Prepare datafile for all Kd values
    bdb3Kd<-bdb5_all_u[,5]
    write.table(bdb3Kd,"BindingDB_Kd.txt",row.names = FALSE)
    
