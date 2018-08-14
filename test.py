import joblib, features_extraction, sys
import whois

def main():
    url="http://www.spit.ac.in"

    features_test=features_extraction.main(url)
    clf = joblib.load('classifier/extratree.pkl')
    features_test = [features_test]
    pred=clf.predict(features_test)
    prob=clf.predict_proba(features_test)
    # print 'Features=', features_test, 'The predicted probability is - ', prob, 'The predicted label is - ', pred
#    print "The probability of this site being a phishing website is ", features_test[0]*100, "%"


    if int(pred[0])==1:
        # print "The website is safe to browse"
        print("SAFE")
    elif int(pred[0])==-1:
        # print "The website has phishing features. DO NOT VISIT!"
        print("PHISHING")

    # print 'Error -', features_test

if __name__=="__main__":
    main()
