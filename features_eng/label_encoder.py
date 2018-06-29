for feature in categorical:
    print(feature)
#     print(f'Transforming {feature}...')
    encoder = LabelEncoder()

    train[feature+"_le"] = encoder.fit_transform(train[feature].astype(str))

    dic = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

#     print(dic['nan'])

    dic['nan']=-999

    test[feature+"_le"]=test[feature].astype(str).map(dic).fillna(-999)

#     encoder.fit(train[feature].append(test[feature]))

# #     train[feature] = encoder.transform(train[feature].astype(str))
# #     test[feature] = encoder.transform(test[feature].astype(str))

# #     train[feature].fillna(-999,inplace=True)
# #     test[feature].fillna(-999,inplace=True)

#     tmp = encoder.transform(train[feature])
