import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import java.util.List;
import java.util.Properties;
import java.io.File;

public class DocumentSplitter {
    public static void main(String[] args) {
        // Accept user-supplied file
        File file = new File(args[0]);

        // Use OCR to extract text from the document
        ITesseract tesseract = new Tesseract();
        String text = tesseract.doOCR(file);

        // Use Stanford NLP to classify and split the documents
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        CoreDocument document = new CoreDocument(text);
        pipeline.annotate(document);

        // Initialize empty lists to store the different document types
        List<CoreSentence> panList = new ArrayList<>();
        List<CoreSentence> aadhaarList = new ArrayList<>();
        List<CoreSentence> bankStatementList = new ArrayList<>();
        List<CoreSentence> itrForm16List = new ArrayList<>();
        List<CoreSentence> customerPhotographList = new ArrayList<>();
        List<CoreSentence> utilityBillList = new ArrayList<>();
        List<CoreSentence> chequeLeafList = new ArrayList<>();
        List<CoreSentence> salarySlipCertificateList = new ArrayList<>();
        List<CoreSentence> drivingLicenseList = new ArrayList<>();
        List<CoreSentence> voterIdList = new ArrayList<>();
        List<CoreSentence> passportList = new ArrayList<>();

        for (CoreSentence sentence : document.sentences()) {
            List<String> posTags = sentence.posTags();
            List<String> lemmas = sentence.lemmas();
            for (int i = 0; i < lemmas.size(); i++) {
                if (lemmas.get(i).equals("PAN") && posTags.get(i).equals("NN")) {
                    panList.add(sentence);
                }
                else if (lemmas.get(i).equals("Aadhaar") && posTags.get(i).equals("NN")) {
                    aadhaarList.add(sentence);
                }
                else if (lemmas.get(i).equals("bank") && posTags.get(i).equals("NN")) {
                    bankStatementList.add(sentence);
                }
                else if (lemmas.get(i).equals("ITR") && posTags.get(
