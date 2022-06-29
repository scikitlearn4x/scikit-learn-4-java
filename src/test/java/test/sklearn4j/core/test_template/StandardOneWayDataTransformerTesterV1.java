package test.sklearn4j.core.test_template;

import ai.sklearn4j.base.TransformerMixin;
import ai.sklearn4j.core.libraries.numpy.NumpyArray;
import ai.sklearn4j.core.packaging.IScikitLearnPackage;
import test.sklearn4j.TestHelper;
import test.sklearn4j.core.test_template.bases.BaseTesterV1;

public class StandardOneWayDataTransformerTesterV1 extends BaseTesterV1 {

    @Override
    protected String getBinaryFilePath(String version) {
        return String.format("%s/%s/%s/%s/%s_%s.skx", TestHelper.TEST_FILES_HOME, version, scikitLearnVersion, pythonVersion, objectName.toLowerCase().replace(" ", "_"), configurationName.replace(" ", "_"));
    }

    @Override
    protected void performUseCaseSpecificTest(IScikitLearnPackage binaryPackage) {
        NumpyArray<Double> transformed = (NumpyArray<Double>) binaryPackage.getExtraValues().get("transformed");
        NumpyArray<Double> raw = (NumpyArray<Double>) binaryPackage.getExtraValues().get("raw");

        TransformerMixin transformerMixin = (TransformerMixin) binaryPackage.getModel("preprocessing_to_test");

        NumpyArray<Double> encoderTransformOutput = (NumpyArray<Double>) transformerMixin.transform(raw);
        TestHelper.assertEqualData(encoderTransformOutput, (double[][]) transformed.getWrapper().getRawArray());
    }

    @Override
    protected void validateExtraValues(IScikitLearnPackage binaryPackage) {

    }
}
