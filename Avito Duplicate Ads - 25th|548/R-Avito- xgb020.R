



#### add number of keys



setwd("D:/NYC Data Science Academy/Kaggle/Avito/")
setwd("/home/rstudio/Dropbox")
# Required Libraries for AWS
install.packages(c("data.table", "xgboost", "caret", "readr", "stringdist", "dplyr", "stringr","drat","textreuse","tm"))

drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type="source")


library(data.table)
library(readr)
library(caret)
library(stringdist)
library(xgboost)
library(dplyr)
library(stringr)
library(textreuse)
library(tm)


install.packages(c("jsonlite", "tidyr"))
library(readr)
library(stringi)
library(jsonlite)





# Read in data
location <- fread("Location.csv")
itemPairsTest  <- fread("ItemPairs_test.csv")
itemPairsTrain <- fread("ItemPairs_train.csv")
itemInfoTest <- read_csv("ItemInfo_test.csv")
itemInfoTrain <- read_csv("ItemInfo_train.csv")
itemInfoTest <- data.table(itemInfoTest)
itemInfoTrain <- data.table(itemInfoTrain)

setkey(location, locationID)
setkey(itemInfoTrain, itemID)
setkey(itemInfoTest, itemID)

#convert to lower case
itemInfoTest$title <- tolower(itemInfoTest$title)
itemInfoTrain$title <- tolower(itemInfoTrain$title)

itemInfoTest$description <- tolower(itemInfoTest$description)
itemInfoTrain$description <- tolower(itemInfoTrain$description)

itemInfoTest$attrsJSON <- tolower(itemInfoTest$attrsJSON)
itemInfoTrain$attrsJSON <- tolower(itemInfoTrain$attrsJSON)

gc()

string_1w <- function(x) {
  sub("(\\w+).*", "\\1", x)
}

string_2w <- function(x) {
  gsub("^((\\w+\\W+){0,1}\\w+).*","\\1",x)
}

# toto = itemInfoTest[1:50,]
# toto1 = toto$attrsJSON
# library(readr)
# library(stringi)
# library(jsonlite)
# listJSON <- sapply(toto1,fromJSON)
# keys <- sapply(listJSON,function(x) return(paste0(unlist(names(x)),collapse=" - ")))
# names(keys) <- NULL
# values <- sapply(listJSON,function(x) return(paste0(unlist(x),collapse=" - ")))
# names(values) <- NULL
# values

string_Json_1 <- function(x) {
  result=ifelse(grepl("вид товара",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}

string_Json_2 <- function(x) {
  result=ifelse(grepl("вид одежды",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}

string_Json_3 <- function(x) {
  result=ifelse(grepl("предмет одежды",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}

string_Json_4 <- function(x) {
  result=ifelse(grepl("размер",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}

string_Json_5 <- function(x) {
  result=ifelse(grepl("тип товара",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}

string_Json_6 <- function(x) {
  result=ifelse(grepl("опыт работы",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}

string_Json_7 <- function(x) {
  result=ifelse(grepl("образование",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}

string_Json_8 <- function(x) {
  result=ifelse(grepl("график работы",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}

string_Json_9 <- function(x) {
  result=ifelse(grepl("сфера деятельности",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}

string_Json_10 <- function(x) {
  result=ifelse(grepl("пол",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(вид[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result
}


string_Json_year <- function(x) {
  result = ifelse(grepl("год",x),gsub(".*(:\")|(.)(\"[,}])","\\2",sub(".*(год[^:]+[(\":\")][^:]+(\",))(.*)", "\\1",x)),NA)
  result = as.numeric(as.character(result))
}

latinLetterCount <- function(x) {
  result = nchar(x)-nchar(gsub("[a-zA-Z]","",x))
  result = as.numeric(as.character(result))
  result
}

figureCount <- function(x) {
  result = nchar(x)-nchar(gsub("[1-9]","",x))
  result = as.numeric(as.character(result))
}

Text_To_Clean_Sentences <- function(text_blob) {
  # swap all sentence ends with code 'ootoo'
  text_blob <- gsub(pattern=';|\\.|!|\\?', x=text_blob, replacement=' ')
  
  # remove all non-alpha text (numbers etc)
  text_blob <- gsub(pattern="[[:punct:]]", x=text_blob, replacement = ' ')
  
  text_blob = gsub(pattern="[\r\n]", replacement = ' ', x=text_blob )
  # force all characters to lower case
  text_blob <- tolower(text_blob)
  
  # remove any small words {size} or {min,max}
  text_blob <- gsub(pattern="\\P{1,3}", x=text_blob, replacement=' ')
  
  text_blob <- removeWords(text_blob, stopwords("russian"))
  text_blob <- removeWords(text_blob, stopwords("english"))
  
  # remove contiguous spaces
  text_blob <- gsub(pattern="\\s+", x=text_blob, replacement=' ')
  
  # split sentences by split code
  sentence_vector <- text_blob
  return (sentence_vector)
}


##json column: to collapse the values as a string.
# attrJson001 = itemInfoTrain$attrsJSON
# attrJson001[is.na(attrJson001)]="{\"unknown\":\"unknown\"}"
# listJSON <- sapply(attrJson001,fromJSON)
# values <- sapply(listJSON,function(x) return(paste0(unlist(x),collapse=" ")))
# names(values) <- NULL
# attrJson001Train <- data.frame(id=itemInfoTrain$itemID, values=values)
# saveRDS(attrJson001Train,"itemInfoTrainJsonValues.rds")
## to load collapsed json values
itemInfoTrainJsonValues = readRDS("itemInfoTrainJsonValues.rds")
itemInfoTrain$attrsJSONcolapsd = as.character(itemInfoTrainJsonValues$values)

# attrJson001 = itemInfoTest$attrsJSON
# attrJson001[is.na(attrJson001)]="{\"unknown\":\"unknown\"}"
# listJSON <- sapply(attrJson001,fromJSON)
# values <- sapply(listJSON,function(x) return(paste0(unlist(x),collapse=" ")))
# names(values) <- NULL
# attrJson001Test <- data.frame(id=itemInfoTest$itemID, values=values)
# saveRDS(attrJson001Test,"itemInfoTestJsonValues.rds")
## to load collapsed json values
itemInfoTestJsonValues = readRDS("itemInfoTestJsonValues.rds")
itemInfoTest$attrsJSONcolapsd = as.character(itemInfoTestJsonValues$values)
rm(itemInfoTrainJsonValues,itemInfoTestJsonValues)







# Drop unused factors
dropAndNumChar <- function(itemInfo){
  itemInfo[, ':=' (ncharTitle = nchar(title),
                   title1w = string_1w(title),
                   title2w = string_2w(title),
                   titlecountvirgule = str_count(title, ","),  
                   titlecountspace = str_count(title, " "),
                   titleLatinLetterCnt = latinLetterCount(title),
                   titleFigureLetterCnt = figureCount(title),
                   
                   
                   
                   ncharDescription = nchar(description),
                   descUptopoint = gsub("[.!\n][\\s\\S]*$", "", description, perl=T),
                   descriptioncountvirgule = str_count(description,"."),   #change , to . in description
                   descriptioncountspace = str_count(description," "),
                   descriptionLatinLetterCnt = latinLetterCount(description),
                   descriptionFigureLetterCnt = figureCount(description),
                   
                   
                   
                   ncharattrsJSON = nchar(attrsJSON),
                   ncharattrsJSONcpsd = nchar(attrsJSONcolapsd),
                   attrsJsoncountvirgule = str_count(attrsJSON,","),
                   attrsJsoncount2points = str_count(attrsJSON,":"),
                   attrsJSON1 = string_Json_1(attrsJSON),
                   attrsJSON2 = string_Json_2(attrsJSON),
                   attrsJSON3 = string_Json_3(attrsJSON),
                   attrsJSON4 = string_Json_4(attrsJSON),
                   attrsJSON5 = string_Json_5(attrsJSON),
                   attrsJSON6 = string_Json_6(attrsJSON),
                   attrsJSON7 = string_Json_7(attrsJSON),
                   attrsJSON8 = string_Json_8(attrsJSON),
                   attrsJSON9 = string_Json_9(attrsJSON),
                   attrsJSON10 = string_Json_10(attrsJSON),
                   attrsJSONyear = string_Json_year(attrsJSON),
                   attrsJSONLatinLetterCnt = latinLetterCount(attrsJSON),
                   attrsJSONFigureLetterCnt = figureCount(attrsJSON),
                   
                   
                   
                   #description = NULL,
                   images_arraycountvirgule = str_count(images_array,",")+1
                   #images_array = NULL#,
                   #attrsJSON = NULL
  )]
}

dropAndNumChar(itemInfoTest)
dropAndNumChar(itemInfoTrain)

gc()


itemInfoTrain$title = Text_To_Clean_Sentences(itemInfoTrain$title)
itemInfoTest$title = Text_To_Clean_Sentences(itemInfoTest$title)

gc()



gc()
# setwd("D:/NYC Data Science Academy/Kaggle/Avito/")
# saveRDS(itemInfoTest,"itemInfoTest1.rds")
# saveRDS(itemInfoTrain,"itemInfoTrain1.rds")

###################################################


# # Read in data
# location <- fread("D:/NYC Data Science Academy/Kaggle/Avito/Location.csv")
# itemPairsTest  <- fread("D:/NYC Data Science Academy/Kaggle/Avito/ItemPairs_test.csv")
# itemPairsTrain <- fread("D:/NYC Data Science Academy/Kaggle/Avito/ItemPairs_train.csv")
# 
# 
# setwd("D:/NYC Data Science Academy/Kaggle/Avito/")
# itemInfoTest = readRDS("itemInfoTest1.rds")
# itemInfoTrain = readRDS("itemInfoTrain1.rds")
# 
# itemPairsTest <- data.table(itemPairsTest)
# itemPairsTrain <- data.table(itemPairsTrain)
# 
# setkey(itemInfoTrain, itemID)
# setkey(itemInfoTest, itemID)

setkey(location, locationID)
setkey(itemInfoTrain, itemID)
setkey(itemInfoTest, itemID)


itemInfoTest$attrsJSONyear = as.numeric(as.character(itemInfoTest$attrsJSONyear))
itemInfoTrain$attrsJSONyear = as.numeric(as.character(itemInfoTrain$attrsJSONyear))

gc()

##############################
#### work the images array to see if there are duplicates:
# library(dplyr)
# library(tidyr)
# itemInfoTrain = data.frame(itemInfoTrain)
# toto11 = itemInfoTrain[,c("itemID","images_array")]
# toto12 = toto11 %>% mutate(images_array = strsplit(as.character(images_array), ",")) %>% unnest(images_array)
# toto12 = toto12[complete.cases(toto11),]
# Traintoto11=toto12
# dupltrain = Traintoto11[duplicated(Traintoto11$images_array) | duplicated(Traintoto11$images_array,fromLast=TRUE),]
# #dupltrain = dupltrain[complete.cases(dupltrain),]
# write.csv(dupltrain,"TrainduplID.csv")
# 
# itemInfoTest=data.frame(itemInfoTest)
# Testtoto11 = itemInfoTest[,c("itemID","images_array")]
# Testtoto11 = Testtoto11 %>% mutate(images_array = strsplit(as.character(images_array), ",")) %>% unnest(images_array)
# dupltest = Testtoto11[duplicated(Testtoto11$images_array) | duplicated(Testtoto11$images_array,fromLast=TRUE),]
# dupltest = dupltest[complete.cases(dupltest),]
# write.csv(dupltest,"TestduplID.csv")
# 
dupltrain = fread("TrainduplID.csv")
dupltest = fread("TestduplID.csv")

itemInfoTrain = itemInfoTrain %>% mutate(imageduplicate = ifelse(itemID %in% dupltrain$itemID,1,0))
itemInfoTest = itemInfoTest %>% mutate(imageduplicate = ifelse(itemID %in% dupltest$itemID,1,0))

sum(itemInfoTrain$imageduplicate)
sum(itemInfoTest$imageduplicate)
rm(dupltrain,dupltest)
gc()
##############################

##############################
#### add category parents

categories <- read_csv("Category.csv")
itemInfoTrain = merge(itemInfoTrain,categories,by.x="categoryID",by.y = "categoryID",all.x = TRUE)
itemInfoTest = merge(itemInfoTest,categories,by.x="categoryID",by.y = "categoryID",all.x = TRUE)
rm(categories)
gc()
##############################

##############################
#### add the hash distance on pictures
head(itemPairsTrain)
dim(itemPairsTrain)
head(itemPairsTest)
dim(itemPairsTest)

uImageTrain=readRDS("/home/rstudio/Dropbox/Avito/ImagesHashForInfoTrain.rds")
uImageTest=readRDS("/home/rstudio/Dropbox/Avito/ImagesHashForInfoTest.rds")
head(uImageTrain)
uImageTrain = ungroup(uImageTrain) %>% arrange(itemID_1,itemID_2)
itemPairsTrain = itemPairsTrain %>% arrange(itemID_1,itemID_2)
head(itemPairsTrain)
tail(itemPairsTrain)

itemPairsTrain=cbind(itemPairsTrain,uImageTrain[,c("minImgMatch","minham","avgham","maxham")])

uImageTest = ungroup(uImageTest) %>% arrange(itemID_1,itemID_2)
itemPairsTest = itemPairsTest %>% arrange(itemID_1,itemID_2)
head(uImageTest)
head(itemPairsTest)
itemPairsTest=cbind(itemPairsTest,uImageTest[,c("minImgMatch","minham","avgham","maxham")])


rm(uImageTrain,uImageTest)
##############################

##############################
#### add the Whash distance on pictures
head(itemPairsTrain)
dim(itemPairsTrain)
head(itemPairsTest)
dim(itemPairsTest)

uImageTrain=readRDS("ImagesWHashForInfoTrain.rds")
uImageTest=readRDS("ImagesWHashForInfoTest.rds")
head(uImageTrain)
uImageTrain = ungroup(uImageTrain) %>% arrange(itemID_1,itemID_2)
itemPairsTrain = itemPairsTrain %>% arrange(itemID_1,itemID_2)
head(itemPairsTrain)
tail(itemPairsTrain)

itemPairsTrain=cbind(itemPairsTrain,uImageTrain[,c("WminImgMatch","minWham","avgWham","maxWham")])

uImageTest = ungroup(uImageTest) %>% arrange(itemID_1,itemID_2)
itemPairsTest = itemPairsTest %>% arrange(itemID_1,itemID_2)
head(uImageTest)
head(itemPairsTest)
itemPairsTest=cbind(itemPairsTest,uImageTest[,c("WminImgMatch","minWham","avgWham","maxWham")])

rm(uImageTrain,uImageTest)
##############################

##############################
### add cos distance on description and title
head(itemPairsTrain)
dim(itemPairsTrain)
head(itemPairsTest)
dim(itemPairsTest)
itemPairsTrain = itemPairsTrain %>% arrange(itemID_1,itemID_2)
itemPairsTest = itemPairsTest %>% arrange(itemID_1,itemID_2)
cos_WtoV_Test = read.csv("test_cos.csv")
cos_WtoV_Test = cos_WtoV_Test %>% arrange(itemID_1,itemID_2)
cos_WtoV_Train = read.csv("train_cos.csv")
cos_WtoV_Train = cos_WtoV_Train %>% arrange(itemID_1,itemID_2)

itemPairsTrain = cbind(itemPairsTrain,cos_WtoV_Train[,c("title_cosine","desc_cosine","title_desc_cosine_1","title_desc_cosine_2")])
itemPairsTest = cbind(itemPairsTest,cos_WtoV_Test[,c("title_cosine","desc_cosine","title_desc_cosine_1","title_desc_cosine_2")])

rm(cos_WtoV_Test,cos_WtoV_Train)
#############################

##############################
### add cos distance on attrs

itemPairsTrain = itemPairsTrain %>% arrange(itemID_1,itemID_2)
itemPairsTest = itemPairsTest %>% arrange(itemID_1,itemID_2)
cos_WtoV_attrs_Test = read.csv("attrs_test_cos.csv")
cos_WtoV_attrs_Test = cos_WtoV_attrs_Test %>% arrange(itemID_1,itemID_2)
cos_WtoV_attrs_Train = read.csv("attrs_train_cos.csv")
cos_WtoV_attrs_Train = cos_WtoV_attrs_Train %>% arrange(itemID_1,itemID_2)

itemPairsTest$attrs_cosine = cos_WtoV_attrs_Test$attrs_cosine
itemPairsTrain$attrs_cosine = cos_WtoV_attrs_Train$attrs_cosine

rm(cos_WtoV_attrs_Test,cos_WtoV_attrs_Train)
#############################

##############################
### add cos distance on attrs

itemPairsTrain = itemPairsTrain %>% arrange(itemID_1,itemID_2)
itemPairsTest = itemPairsTest %>% arrange(itemID_1,itemID_2)
Common_keys_Json_Test = read.csv("key_in_common_test.csv")
Common_keys_Json_Test = Common_keys_Json_Test %>% arrange(itemID_1,itemID_2)
Common_keys_Json_Train = read.csv("key_in_common_train.csv")
Common_keys_Json_Train = Common_keys_Json_Train %>% arrange(itemID_1,itemID_2)

itemPairsTest$attrs_com_keys = Common_keys_Json_Test$key_in_common
itemPairsTrain$attrs_com_keys = Common_keys_Json_Train$key_in_common


rm(Common_keys_Json_Test,Common_keys_Json_Train)
#############################



itemInfoTrain = data.table(itemInfoTrain)
itemInfoTest = data.table(itemInfoTest)
itemPairsTrain = data.table(itemPairsTrain)
itemPairsTest = data.table(itemPairsTest)

setkey(location, locationID)
setkey(itemInfoTrain, itemID)
setkey(itemInfoTest, itemID)


# Merge
mergeInfo <- function(itemPairs, itemInfo){
  # merge on itemID_1
  setkey(itemPairs, itemID_1)
  itemPairs <- itemInfo[itemPairs]
  setnames(itemPairs, names(itemInfo), paste0(names(itemInfo), "_1"))
  # merge on itemID_2
  setkey(itemPairs, itemID_2)
  itemPairs <- itemInfo[itemPairs]
  setnames(itemPairs, names(itemInfo), paste0(names(itemInfo), "_2"))
  # merge on locationID_1
  setkey(itemPairs, locationID_1)
  itemPairs <- location[itemPairs]
  setnames(itemPairs, names(location), paste0(names(location), "_1"))
  # merge on locationID_2
  setkey(itemPairs, locationID_2)
  itemPairs <- location[itemPairs]
  setnames(itemPairs, names(location), paste0(names(location), "_2"))
  return(itemPairs)
}

itemPairsTrain <- mergeInfo(itemPairsTrain, itemInfoTrain)
itemPairsTest <- mergeInfo(itemPairsTest, itemInfoTest)

#rm(list=c("itemInfoTest", "itemInfoTrain", "location"))




# Create features
matchPair <- function(x, y){
  ifelse(is.na(x), ifelse(is.na(y), 3, 2), ifelse(is.na(y), 2, ifelse(x==y, 1, 4)))
}

matchPairNumber <- function(x, y){
  ifelse(is.na(x), ifelse(is.na(y), 99999, 9999), ifelse(is.na(y), 9999, ifelse(x==y, 0, abs(x-y))))
}

# 
# itemPairsTest$titleyear_2 = as.numeric(as.character(itemPairsTest$titleyear_2))
# itemPairsTrain$titleyear_2 = as.numeric(as.character(itemPairsTrain$titleyear_2))
itemPairsTest$attrsJSONyear_2 = as.numeric(as.character(itemPairsTest$attrsJSONyear_2))
itemPairsTrain$attrsJSONyear_2 = as.numeric(as.character(itemPairsTrain$attrsJSONyear_2))
# itemPairsTest$titleyear_1 = as.numeric(as.character(itemPairsTest$titleyear_1))
# itemPairsTrain$titleyear_1 = as.numeric(as.character(itemPairsTrain$titleyear_1))
itemPairsTest$attrsJSONyear_1 = as.numeric(as.character(itemPairsTest$attrsJSONyear_1))
itemPairsTrain$attrsJSONyear_1 = as.numeric(as.character(itemPairsTrain$attrsJSONyear_1))









itemPairsTrain$titleclean_1 = strsplit(itemPairsTrain$title_1," ") 
itemPairsTest$titleclean_1 = strsplit(itemPairsTest$title_1," ") 
itemPairsTrain$titleclean_2 = strsplit(itemPairsTrain$title_2," ")
itemPairsTest$titleclean_2 = strsplit(itemPairsTest$title_2," ")

itemPairsTrain$titlejacsim = mapply(jaccard_similarity,itemPairsTrain$titleclean_1, itemPairsTrain$titleclean_2)
itemPairsTrain$titlejacdissim = mapply(jaccard_dissimilarity,itemPairsTrain$titleclean_1, itemPairsTrain$titleclean_2)
itemPairsTrain$titlejacbagsim = mapply(jaccard_bag_similarity,itemPairsTrain$titleclean_1, itemPairsTrain$titleclean_2)
itemPairsTrain$titlejacratio = mapply(ratio_of_matches,itemPairsTrain$titleclean_1, itemPairsTrain$titleclean_2)

itemPairsTest$titlejacsim = mapply(jaccard_similarity,itemPairsTest$titleclean_1, itemPairsTest$titleclean_2)
itemPairsTest$titlejacdissim = mapply(jaccard_dissimilarity,itemPairsTest$titleclean_1, itemPairsTest$titleclean_2)
itemPairsTest$titlejacbagsim = mapply(jaccard_bag_similarity,itemPairsTest$titleclean_1, itemPairsTest$titleclean_2)
itemPairsTest$titlejacratio = mapply(ratio_of_matches,itemPairsTest$titleclean_1, itemPairsTest$titleclean_2)

itemPairsTrain$titleclean_1 = NULL
itemPairsTest$titleclean_1 = NULL 
itemPairsTrain$titleclean_2 = NULL
itemPairsTest$titleclean_2 = NULL

#setwd("D:/NYC Data Science Academy/Kaggle/Avito/")
saveRDS(itemPairsTest, "itemPairsTestfin0194.rds")
saveRDS(itemPairsTrain,"itemPairsTrainfin0194.rds")
# 

#reload from here and include the damned jaccards.
# 
itemPairsTest = readRDS("itemPairsTestfin0194.rds")
itemPairsTrain = readRDS("itemPairsTrainfin0194.rds")




matchPair <- function(x, y){
  ifelse(is.na(x), ifelse(is.na(y), 3, 2), ifelse(is.na(y), 2, ifelse(x==y, 1, 4)))
}

matchPairNumber <- function(x, y){
  ifelse(is.na(x), ifelse(is.na(y), 99999, 9999), ifelse(is.na(y), 9999, ifelse(x==y, 0, abs(x-y))))
}

matchPairPercent <- function(x, y){
  ifelse(is.na(x), ifelse(is.na(y), 99999, 9999), ifelse(is.na(y), 9999, ifelse(x==y, 0, abs(x-y)/abs(x+y))))
}



createFeatures <- function(itemPairs){
  itemPairs[, ':=' (
                    
                    # a revoir avec json concatenate sur attr
                    # a revoir avec espace et virgule match...
                    # Full Damerau-Levenshtein distance.
                    
                    # a revoir --- pas tres clair d avoir une abs diff on location...
                    locationMatchN = matchPairNumber(locationID_1, locationID_2),
                    #locationID_1 = NULL,
                    #locationID_2 = NULL,
                    regionMatchN = matchPairNumber(regionID_1, regionID_2),
                    #regionID_1 = NULL,
                    #regionID_2 = NULL,
                    metroMatchN = matchPairNumber(metroID_1, metroID_2),
                    #metroID_1 = NULL,
                    #metroID_2 = NULL,
                    
                    

                    images_arrayVirgMatch = matchPairPercent(images_arraycountvirgule_1, images_arraycountvirgule_2),
                    #images_arraycountvirgule_1 = NULL,
                    #images_arraycountvirgule_2 = NULL,

                    #categoryMatch = matchPairNumber(categoryID_1, categoryID_2),
                    categoryID_1 = NULL,
                    #categoryID_2 = NULL,
                    #parentCategoryMatch = matchPairNumber(parentCategoryID_1, parentCategoryID_2),
                    parentCategoryID_1 = NULL,
                    #categoryID_2 = NULL,
                    
                    
                    priceMatch = matchPairNumber(price_1, price_2),
                    priceDiff = pmax(price_1/price_2, price_2/price_1),
                    priceMin = pmin(price_1, price_2, na.rm=TRUE),
                    priceMax = pmax(price_1, price_2, na.rm=TRUE),
                    pricereldist = abs(price_1-price_2)/(abs(price_1)+abs(price_2)),
                    #price_1 = NULL,
                    #price_2 = NULL,
                    
                    titlecountvirgule = matchPairPercent(titlecountvirgule_1,titlecountvirgule_2), 
                    #titlecountvirgule_1 = NULL,
                    titlecountspace = matchPairPercent(titlecountspace_1,titlecountspace_2),
                    #titlecountspace_1 = NULL,
                    descriptioncountvirgule = matchPairPercent(descriptioncountvirgule_1,descriptioncountvirgule_2),
                    #descriptioncountvirgule_1 = NULL,
                    descriptioncountspace = matchPairPercent(descriptioncountspace_1,descriptioncountspace_2),
                    #descriptioncountspace_1 = NULL,
                    
                    
                    titleStringJw = stringdist(title_1, title_2, method = "jw"),
                    titleStringDl = stringdist(title_1, title_2, method = "dl") /pmax(ncharTitle_1, ncharTitle_2, na.rm=TRUE),
                    titleStringLcs = (stringdist(title_1, title_2, method = "lcs") /pmax(ncharTitle_1, ncharTitle_2, na.rm=TRUE)),
                     title1StartsWithTitle2 = as.numeric(substr(title_1, 1, nchar(title_2)) == title_2),
                     title2StartsWithTitle1 = as.numeric(substr(title_2, 1, nchar(title_1)) == title_1),
                    title_1 = NULL,
                    title_2 = NULL,
                    
                    
                     title1wStringJw = stringdist(title1w_1, title1w_2, method = "jw"),
                    title1wStringDl = stringdist(title1w_1, title1w_2, method = "dl")/pmax(nchar(title1w_1), nchar(title1w_2), na.rm=TRUE),
                     title1wStringLcs = stringdist(title1w_1, title1w_2, method = "lcs")/pmax(nchar(title1w_1), nchar(title1w_2), na.rm=TRUE),
                    title1w_1 = NULL,
                    title1w_2 = NULL,
                    
                     title2wStringJw = stringdist(title2w_1, title2w_2, method = "jw"),
                    title2wStringDl = stringdist(title2w_1, title2w_2, method = "dl")/pmax(nchar(title2w_1), nchar(title2w_2), na.rm=TRUE),
                     title2wStringLcs = stringdist(title2w_1, title2w_2, method = "lcs")/pmax(nchar(title2w_1), nchar(title2w_2), na.rm=TRUE),
                    title2w_1 = NULL,
                    title2w_2 = NULL,
                    
                    # voir si any impact
                     imageduplicatMatch = matchPairPercent(imageduplicate_1, imageduplicate_2),
                    #imageduplicate_1=NULL,
                    #imageduplicate_2=NULL,
                    
                    
                     titleCharDiff = pmax(ncharTitle_1/ncharTitle_2, ncharTitle_2/ncharTitle_1),
                     titleCharMin = pmin(ncharTitle_1, ncharTitle_2, na.rm=TRUE),
                     titleCharMax = pmax(ncharTitle_1, ncharTitle_2, na.rm=TRUE),
                     titlereldist = abs(ncharTitle_1-ncharTitle_2)/(abs(ncharTitle_1)+abs(ncharTitle_2)),
                     #ncharTitle_1 = NULL,
                     #ncharTitle_2 = NULL,
                    
                    
                    
                     descriptionCharDiff = pmax(ncharDescription_1/ncharDescription_2, ncharDescription_2/ncharDescription_1),
                     descriptionCharMin = pmin(ncharDescription_1, ncharDescription_2, na.rm=TRUE),
                     descriptionCharMax = pmax(ncharDescription_1, ncharDescription_2, na.rm=TRUE),
                     descriptionreldist = abs(ncharDescription_1-ncharDescription_2)/(abs(ncharDescription_1)+abs(ncharDescription_2)),
                     #ncharDescription_1 = NULL,
                     #ncharDescription_2 = NULL,
                     
                     descriptionStringJw = stringdist(description_1, description_2, method = "jw"),
                    descriptionStringDl = stringdist(description_1, description_2, method = "dl") /pmax(ncharDescription_1, ncharDescription_2, na.rm=TRUE),
                     descriptionStringLcs = stringdist(description_1, description_2, method = "lcs") /pmax(ncharDescription_1, ncharDescription_2, na.rm=TRUE),
                    description_1 = NULL,
                    description_2 = NULL,
                    
                    desc1Title2StringJw = stringdist(description_1, title_2, method = "jw"),
                    desc2Title1StringJw = stringdist(title_1, description_2, method = "jw"),
            
                    #descUptopointStringDist = stringdist(descUptopoint_1, descUptopoint_2, method = "jw"),
                    #descUptopointStringDist2 = stringdist(descUptopoint_1, descUptopoint_2, method = "dl") ,
                    #descUptopointStringDist3 = stringdist(descUptopoint_1, descUptopoint_2, method = "lcs")
                    descUptopoint_1 = NULL,
                    descUptopoint_2 = NULL,
                    
                     attrsJSONCharDiff = pmax(ncharattrsJSON_1/ncharattrsJSON_2, ncharattrsJSON_2/ncharattrsJSON_1),
                     attrsJSONCharMin = pmin(ncharattrsJSON_1, ncharattrsJSON_2, na.rm=TRUE),
                     attrsJSONCharMax = pmax(ncharattrsJSON_1, ncharattrsJSON_2, na.rm=TRUE),
                     attrsJSONreldist = abs(ncharattrsJSON_1-ncharDescription_2)/(abs(ncharattrsJSON_1)+abs(ncharattrsJSON_2)),
                     #ncharattrsJSON_1 = NULL,
                     #ncharattrsJSON_2 = NULL,
                     ncharattrsJSONcpsdCharDiff = pmax(ncharattrsJSONcpsd_1/ncharattrsJSONcpsd_2, ncharattrsJSONcpsd_2/ncharattrsJSONcpsd_1),
                    
                     attrsJSONStringJw = stringdist(attrsJSON_1, attrsJSON_2, method = "jw"),
                     attrsJSONStringDl = stringdist(attrsJSON_1, attrsJSON_2, method = "dl")/pmax(ncharattrsJSON_1, ncharattrsJSON_2, na.rm=TRUE),
                     attrsJSONStringLcs = stringdist(attrsJSON_1, attrsJSON_2, method = "lcs")/pmax(ncharattrsJSON_1, ncharattrsJSON_2, na.rm=TRUE),
                     attrsJSON_1 = NULL,
                     attrsJSON_2 = NULL
                    
  )]
  
  #itemPairs[, ':=' (#priceDiff = ifelse(is.na(priceDiff), 9999, priceDiff),
    #priceMin = ifelse(is.na(priceMin), 9999, priceMin),
    #priceMax = ifelse(is.na(priceMax), 9999, priceMax),
    #titleStringDist = ifelse(is.na(titleStringDist), 9999, titleStringDist),
    #titleStringDist2 = ifelse(is.na(titleStringDist2) | titleStringDist2 == Inf, 9999, titleStringDist2))]
}

createFeatures(itemPairsTest)
createFeatures(itemPairsTrain)

gc()





createFeatures2 <- function(itemPairs){
  itemPairs[, ':=' (
    
    
#    titlecllist descllist attrcllist
    
    #titlejacsim = jaccard_similarity(title_1, titlecllist_2),
    #titlejacdissim = jaccard_dissimilarity(titlecllist_1, titlecllist_2),
    #titlejacbagsim = jaccard_bag_similarity(titlecllist_1, titlecllist_2),
    #titlejacratio = ratio_of_matches(titlecllist_1, titlecllist_2),
    # 
    # descjacsim = jaccard_similarity(descllist_1, descllist_2),
    # descjacdissim = jaccard_dissimilarity(descllist_1, descllist_2),
    # descjacbagsim = jaccard_bag_similarity(descllist_1, descllist_2),
    # descjacratio = ratio_of_matches(descllist_1, descllist_2),
    # 
    # attrjacsim = jaccard_similarity(attrcllist_1, attrcllist_2),
    # attrjacdissim = jaccard_dissimilarity(attrcllist_1, attrcllist_2),
    # attrjacbagsim = jaccard_bag_similarity(attrcllist_1, attrcllist_2),
    # attrjacratio = ratio_of_matches(attrcllist_1, attrcllist_2),
    # 
    # titlecllist_1 = NULL,
    # titlecllist_2 = NULL,
    # descllist_1 = NULL,
    # descllist_2 = NULL,
    # attrcllist_1 = NULL,
    # attrcllist_2 = NULL,
    
    
    #added
    attrsJSON1Match = matchPair(attrsJSON1_1, attrsJSON1_2),
    attrsJSON1_1 = NULL,
    attrsJSON1_2 = NULL,
    
    attrsJSON2Match = matchPair(attrsJSON2_1, attrsJSON2_2),
    attrsJSON2_1 = NULL,
    attrsJSON2_2 = NULL,
    
    attrsJSON3Match = matchPair(attrsJSON3_1, attrsJSON3_2),
    attrsJSON3_1 = NULL,
    attrsJSON3_2 = NULL,
    attrsJSON4Match = matchPair(attrsJSON4_1, attrsJSON4_2),
    attrsJSON4_1 = NULL,
    attrsJSON4_2 = NULL,
    attrsJSON5Match = matchPair(attrsJSON5_1, attrsJSON5_2),
    attrsJSON5_1 = NULL,
    attrsJSON5_2 = NULL,
    attrsJSON6Match = matchPair(attrsJSON6_1, attrsJSON6_2),
    attrsJSON6_1 = NULL,
    attrsJSON6_2 = NULL,
    attrsJSON7Match = matchPair(attrsJSON7_1, attrsJSON7_2),
    attrsJSON7_1 = NULL,
    attrsJSON7_2 = NULL,
    attrsJSON8Match = matchPair(attrsJSON8_1, attrsJSON8_2),
    attrsJSON8_1 = NULL,
    attrsJSON8_2 = NULL,
    attrsJSON9Match = matchPair(attrsJSON9_1, attrsJSON9_2),
    attrsJSON9_1 = NULL,
    attrsJSON9_2 = NULL,
    attrsJSON10Match = matchPair(attrsJSON10_1, attrsJSON10_2),
    attrsJSON10_1 = NULL,
    attrsJSON10_2 = NULL,
    
    attrsJSONMatch = matchPairNumber(attrsJSONyear_1, attrsJSONyear_2),
    
    
    distance = sqrt((lat_1-lat_2)^2+(lon_1-lon_2)^2),
    #lat_1 = NULL,
    #lat_2 = NULL,
    #lon_1 = NULL,
    #lon_2 = NULL,
    #itemID_1 = NULL
    #itemID_2 = NULL
    titleLatinMatch = matchPairPercent(titleLatinLetterCnt_1, titleLatinLetterCnt_2),
    descLatinMatch = matchPairPercent(descriptionLatinLetterCnt_1, descriptionLatinLetterCnt_2),
    attrsLatinMatch = matchPairPercent(attrsJSONLatinLetterCnt_1, attrsJSONLatinLetterCnt_2),
    
    titleFigureMatch = matchPairPercent(titleFigureLetterCnt_1, titleFigureLetterCnt_2),
    descFigureMatch = matchPairPercent(descriptionFigureLetterCnt_1, descriptionFigureLetterCnt_2),
    attrsFigureMatch = matchPairNumber(attrsJSONFigureLetterCnt_1, attrsJSONFigureLetterCnt_2)
    
    
    
  )]
  
  #itemPairs[, ':=' (#priceDiff = ifelse(is.na(priceDiff), 9999, priceDiff),
  #priceMin = ifelse(is.na(priceMin), 9999, priceMin),
  #priceMax = ifelse(is.na(priceMax), 9999, priceMax),
  #titleStringDist = ifelse(is.na(titleStringDist), 9999, titleStringDist),
  #titleStringDist2 = ifelse(is.na(titleStringDist2) | titleStringDist2 == Inf, 9999, titleStringDist2))]
}

createFeatures2(itemPairsTest)
createFeatures2(itemPairsTrain)

gc()












#setwd("D:/NYC Data Science Academy/Kaggle/Avito/")
saveRDS(itemPairsTest,"itemPairsTestfin020.rds")
saveRDS(itemPairsTrain,"itemPairsTrainfin020.rds")
# 
# 
# itemPairsTest  = readRDS("itemPairsTestfin017.rds")
# itemPairsTrain = readRDS("itemPairsTrainfin017.rds")


itemPairsTest$images_array_1=NULL
itemPairsTest$images_array_2=NULL
itemPairsTrain$images_array_1=NULL
itemPairsTrain$images_array_2=NULL

itemPairsTest$attrsJSONcolapsd_1=NULL
itemPairsTest$attrsJSONcolapsd_2=NULL
itemPairsTrain$attrsJSONcolapsd_1=NULL
itemPairsTrain$attrsJSONcolapsd_2=NULL


itemPairsTest <- read_csv("itemPairsTestfin020_lin_et_001.csv")
itemPairsTrain <- read_csv("itemPairsTrainfin020_lin_et_001.csv")





library(xgboost)



modelVars <- names(itemPairsTrain)[which(!(names(itemPairsTrain) %in% c("id", "isDuplicate", "generationMethod", "foldId")))]

itemPairsTest <- data.frame(itemPairsTest)
itemPairsTrain <- data.frame(itemPairsTrain)

# itemPairsTest$titleyear_2 = as.numeric(as.character(itemPairsTest$titleyear_2))
# itemPairsTrain$titleyear_2 = as.numeric(as.character(itemPairsTrain$titleyear_2))
# itemPairsTest$attrsJSONyear_2 = as.numeric(as.character(itemPairsTest$attrsJSONyear_2))
# itemPairsTrain$attrsJSONyear_2 = as.numeric(as.character(itemPairsTrain$attrsJSONyear_2))
# itemPairsTest$titleyear_1 = as.numeric(as.character(itemPairsTest$titleyear_1))
# itemPairsTrain$titleyear_1 = as.numeric(as.character(itemPairsTrain$titleyear_1))
# itemPairsTest$attrsJSONyear_1 = as.numeric(as.character(itemPairsTest$attrsJSONyear_1))
# itemPairsTrain$attrsJSONyear_1 = as.numeric(as.character(itemPairsTrain$attrsJSONyear_1))

set.seed(1984)
itemPairsTrain2 <- itemPairsTrain[sample(nrow(itemPairsTrain), 250000), ]


w1 = itemPairsTrain2[, "generationMethod"]
a=1
b=1
c=1
w1[w1==1]=a
w1[w1==3]=b
w1[w1==2]=c
# for (j in 1:79) {
#   itemPairsTrain[,j]=as.numeric(as.character(itemPairsTrain[,j]))
# }
# 
# for (j in 1:78) {
#   itemPairsTest[,j]=as.numeric(as.character(itemPairsTest[,j]))
# }


# Matrix
dtraincv <- xgb.DMatrix(as.matrix(itemPairsTrain2[, modelVars]), label=itemPairsTrain2$isDuplicate,missing=NA,weight=w1)


# xgboost cross-validated
set.seed(2017)
maxTrees <- 3000
shrinkage <- 0.1
#gamma <- 2
depth <- 10
minChildWeight <- 300
colSample <- .5
colSampletree <- 1
subSample <- .7
earlyStopRound <- 50

set.seed(2017)
xgbCV <- xgb.cv(params=list(max_depth=depth,
                            eta=shrinkage,
                            #gamma=gamma,
                            #sample_type="weighted",
                            #normalize_type="forest",
                            colsample_bylevel=colSample,
                            colsample_bytree=colSampletree,
                            min_child_weight=minChildWeight,
                            subsample=subSample,
                            objective="binary:logistic"),
                data=dtraincv,
                nrounds=maxTrees,
                print.every.n = 50,
                eval_metric ="auc",
                nfold=5,
                missing=NA,
                #stratified=TRUE,
                early.stop.round=earlyStopRound)

numTrees <- min(which(xgbCV$test.auc.mean==max(xgbCV$test.auc.mean)))
numTrees
max(xgbCV$test.auc.mean)

gc()


w2 = itemPairsTrain[, "generationMethod"]
w2[w2==1]=a
w2[w2==3]=b
w2[w2==2]=c

set.seed(2017)
dtrain <- xgb.DMatrix(as.matrix(itemPairsTrain[, modelVars]), label=itemPairsTrain$isDuplicate,missing=NA,weight=w2)

xgbResult <- xgb.train(params=list(max_depth=depth,
                                   eta=shrinkage,
                                   #gamma=gamma,
                                   colsample_bylevel=colSample,
                                   colsample_bytree=colSampletree,
                                   subsample=subSample,
                                   min_child_weight=minChildWeight),
                       data=dtrain, 
                       nrounds=numTrees,
                       #print.every.n = 100,
                       missing=NA,
                       objective="binary:logistic",
                       #stratified=TRUE,
                       eval_metric="auc")


dtest <- xgb.DMatrix(as.matrix(itemPairsTest[, modelVars]),missing=NA)
testPreds <- predict(xgbResult, dtest)

#setwd("D:/NYC Data Science Academy/Kaggle/Avito/")
submission <- data.frame(id=itemPairsTest$id, probability=testPreds)
write.csv(submission, file="submission033-xgb020.csv",row.names=FALSE)

trainPreds <- predict(xgbResult, dtrain)

setwd("C:/Users/xcapdepon/Documents/Capdepon personal/Avito")
submission <- data.frame(id=itemPairsTrain$isDuplicate, probability=trainPreds)
write.csv(submission, file="trainsubmission008-reg.csv",row.names=FALSE)



# [0]	train-auc:0.828432+0.001259	test-auc:0.814352+0.001878
# [50]	train-auc:0.890305+0.000422	test-auc:0.858754+0.002065
# [100]	train-auc:0.908929+0.001007	test-auc:0.866627+0.001678
# [150]	train-auc:0.921560+0.000810	test-auc:0.869814+0.001683
# [200]	train-auc:0.931180+0.001032	test-auc:0.871548+0.001806
# [250]	train-auc:0.938889+0.000977	test-auc:0.872618+0.001627
# [300]	train-auc:0.945826+0.000580	test-auc:0.873217+0.001552
# [350]	train-auc:0.951903+0.000390	test-auc:0.873614+0.001571
# [400]	train-auc:0.957230+0.000541	test-auc:0.874023+0.001637
# [450]	train-auc:0.961939+0.000371	test-auc:0.874260+0.001670
# [500]	train-auc:0.966122+0.000659	test-auc:0.874384+0.001696
# [550]	train-auc:0.969860+0.000541	test-auc:0.874461+0.001682
# [600]	train-auc:0.973458+0.000437	test-auc:0.874541+0.001556
# Stopping. Best iteration: 595











################################################################



set.seed(2017)
xgbCV <- xgb.cv(booster="gblinear",
                params=list(max_depth=depth,
                            eta=shrinkage,
                            #gamma=gamma,
                            #sample_type="weighted",
                            #normalize_type="forest",
                            lambda = .5,
                            alpha = .5,
                            subsample=subSample,
                            objective="binary:logistic"),
                data=dtraincv,
                nrounds=maxTrees,
                print.every.n = 50,
                eval_metric ="auc",
                nfold=5,
                missing=NA,
                #stratified=TRUE,
                early.stop.round=earlyStopRound)

numTrees <- min(which(xgbCV$test.auc.mean==max(xgbCV$test.auc.mean)))
numTrees
max(xgbCV$test.auc.mean)


#####################################



train_reg <- fread("trainsubmission008-reg.csv")
train_log <- fread("trainsubmission007.csv")

test_reg <- fread("submission008-reg.csv")
test_log <- fread("submission007.csv")

test_reg$probability = ifelse(test_reg$probability<0.0,0.0,ifelse(test_reg$probability>1,1,test_reg$probability))
max(test_reg$probability)
min(test_reg$probability)

mean(test_reg$probability)
mean(test_log$probability)

write.csv(test_reg, file="submission008-regc.csv",row.names=FALSE)



mean(train_reg$probability)
mean(train_log$probability)


library(pROC)
library(ROCR)

pred <- ROCR::prediction(train_reg$id,ifelse(train_reg$probability<=.5,0,1))
auc <- ROCR::performance(pred,"auc")
auc

testpow1 = (test_reg$probability^1 + test_log$probability^1)/2
testpow2 = (test_reg$probability^2 + test_log$probability^2)/2
testpow4 = (test_reg$probability^4 + test_log$probability^4)/2


# power on train
i=3
trainpow = (train_reg$probability^i + train_log$probability^i)/2
pred <- ROCR::prediction(train_reg$id,ifelse(trainpow<=.5,0,1))
auc <- ROCR::performance(pred,"auc")
auc
# rank on train
trainrank = (percent_rank(train_reg$probability) + percent_rank(train_log$probability))/2
pred <- ROCR::prediction(train_reg$id,ifelse(trainrank<=.5,0,1))
auc <- ROCR::performance(pred,"auc")
auc

testpow8 = (test_reg$probability^8 + test_log$probability^8)/2


testpow1_sub = test_reg
testpow1_sub$probability=testpow1
write.csv(testpow1_sub, file="submission009-007 008 pow1.csv",row.names=FALSE)


testpow2_sub = test_reg
testpow2_sub$probability=testpow2
write.csv(testpow2_sub, file="submission009-007 008 pow2.csv",row.names=FALSE)


testpow4_sub = test_reg
testpow4_sub$probability=testpow4
write.csv(testpow4_sub, file="submission010-007 008 pow4.csv",row.names=FALSE)


