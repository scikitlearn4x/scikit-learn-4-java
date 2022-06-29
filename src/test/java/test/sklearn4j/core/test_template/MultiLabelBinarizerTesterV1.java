package test.sklearn4j.core.test_template;

import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.libraries.numpy.NumpyArrayFactory;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import ai.sklearn4j.preprocessing.label.MultiLabelBinarizer;
import org.junit.jupiter.api.Assertions;
import test.sklearn4j.TestHelper;
import test.sklearn4j.core.test_template.bases.BaseTesterV1;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class MultiLabelBinarizerTesterV1 extends BaseTesterV1 {
    public MultiLabelBinarizer objectUnderTest;

    @Override
    protected String getBinaryFilePath(String version) {
        return String.format("%s/%s/%s/%s/%s_%s.skx", TestHelper.TEST_FILES_HOME, version, scikitLearnVersion, pythonVersion, objectName.toLowerCase().replace(" ", "_"), configurationName.replace(" ", "_"));
    }

    @Override
    protected void performUseCaseSpecificTest(IScikitLearnPackage binaryPackage) {
        objectUnderTest = (MultiLabelBinarizer) binaryPackage.getModel("preprocessing_to_test");

        long[][] transformed = (long[][]) ((NumpyArray) binaryPackage.getExtraValues().get("transformed")).getWrapper().getRawArray();
        Object _raw = binaryPackage.getExtraValues().get("raw");
        List<Set> raw = null;

        if (_raw instanceof ArrayList) {
            Object value = ((List) _raw).get(0);
            List<Set<Object>> tmp = new ArrayList<>();

            for (Object l : ((List) _raw)) {
                ArrayList<Object> list = (ArrayList<Object>) l;
                Set<Object> s = list.stream().collect(Collectors.toSet());
                tmp.add(s);
            }

            MultiLabelBinarizer encoder = (MultiLabelBinarizer) binaryPackage.getModel("preprocessing_to_test");

            NumpyArray<Long> encoderTransformOutput = encoder.transform(tmp);
            TestHelper.assertEqualData(encoderTransformOutput, transformed);

            List<Set<Object>> encoderInverseTransformOutput = encoder.inverseTransform(NumpyArrayFactory.from(transformed));
            assertEqualData(encoderInverseTransformOutput, tmp);
        } else {
            Assertions.fail();
        }
    }

    @Override
    protected void validateExtraValues(IScikitLearnPackage binaryPackage) {

    }

    public static void assertEqualData(List<Set<Object>> l1, List<Set<Object>> l2) {
        Assertions.assertEquals(l1.size(), l2.size());

        for (int i = 0; i < l1.size(); i++) {
            Assertions.assertEquals(l1.get(i), l2.get(i));
        }
    }
}
