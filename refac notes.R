
missvalues_visual <- 
  df  %>%
  summarise_all(list(~is.na(.)))%>%
  pivot_longer(everything(),
               names_to = "variables", values_to="missing") %>%
  count(variables, missing) %>%
  ggplot(aes(y=variables,x=n,fill=missing))+
  geom_col()+
  scale_fill_manual(values=c("skyblue3","gold"))+
  theme(axis.title.y=element_blank())
missvalues_visual
options(repr.plot.width = 14, repr.plot.height = 16)



for (i in names(df[, null_cols])) {
  # Grouping of variables dependent on the presence of a basement
  if (str_detect(i, "Bsmt") == TRUE) {
    df[, i][is.na(df[, i])] <- 'No Basement'
    
    # Grouping of variables dependent on the presence of a garage
  } else if (str_detect(i, "Garage") == TRUE) {
    if (i == 'GarageYrBlt'){
      df[, i][is.na(df[, i])] <- 0
    } else {
      df[, i][is.na(df[, i])] <- 'No Garage'
    }
    
  }
}
